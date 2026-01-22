import itertools
import logging
import warnings
import os_ken.base.app_manager
from os_ken.lib import hub
from os_ken import utils
from os_ken.controller import ofp_event
from os_ken.controller.controller import OpenFlowController
from os_ken.controller.handler import set_ev_handler
from os_ken.controller.handler import HANDSHAKE_DISPATCHER, CONFIG_DISPATCHER,\
from os_ken.ofproto import ofproto_parser
class OFPHandler(os_ken.base.app_manager.OSKenApp):

    def __init__(self, *args, **kwargs):
        super(OFPHandler, self).__init__(*args, **kwargs)
        self.name = ofp_event.NAME
        self.controller = None

    def start(self):
        super(OFPHandler, self).start()
        self.controller = OpenFlowController()
        return hub.spawn(self.controller)

    def _hello_failed(self, datapath, error_desc):
        self.logger.error('%s on datapath %s', error_desc, datapath.address)
        error_msg = datapath.ofproto_parser.OFPErrorMsg(datapath=datapath, type_=datapath.ofproto.OFPET_HELLO_FAILED, code=datapath.ofproto.OFPHFC_INCOMPATIBLE, data=error_desc)
        datapath.send_msg(error_msg, close_socket=True)

    @set_ev_handler(ofp_event.EventOFPHello, HANDSHAKE_DISPATCHER)
    def hello_handler(self, ev):
        self.logger.debug('hello ev %s', ev)
        msg = ev.msg
        datapath = msg.datapath
        elements = getattr(msg, 'elements', None)
        if elements:
            switch_versions = set()
            for version in itertools.chain.from_iterable((element.versions for element in elements)):
                switch_versions.add(version)
            usable_versions = switch_versions & set(datapath.supported_ofp_version)
            negotiated_versions = set((version for version in switch_versions if version <= max(datapath.supported_ofp_version)))
            if negotiated_versions and (not usable_versions):
                error_desc = 'no compatible version found: switch versions %s controller version 0x%x, the negotiated version is 0x%x, but no usable version found. If possible, set the switch to use one of OF version %s' % (switch_versions, max(datapath.supported_ofp_version), max(negotiated_versions), sorted(datapath.supported_ofp_version))
                self._hello_failed(datapath, error_desc)
                return
            if negotiated_versions and usable_versions and (max(negotiated_versions) != max(usable_versions)):
                error_desc = 'no compatible version found: switch versions 0x%x controller version 0x%x, the negotiated version is %s but found usable %s. If possible, set the switch to use one of OF version %s' % (max(switch_versions), max(datapath.supported_ofp_version), sorted(negotiated_versions), sorted(usable_versions), sorted(usable_versions))
                self._hello_failed(datapath, error_desc)
                return
        else:
            usable_versions = set((version for version in datapath.supported_ofp_version if version <= msg.version))
            if usable_versions and max(usable_versions) != min(msg.version, datapath.ofproto.OFP_VERSION):
                version = max(usable_versions)
                error_desc = 'no compatible version found: switch 0x%x controller 0x%x, but found usable 0x%x. If possible, set the switch to use OF version 0x%x' % (msg.version, datapath.ofproto.OFP_VERSION, version, version)
                self._hello_failed(datapath, error_desc)
                return
        if not usable_versions:
            error_desc = 'unsupported version 0x%x. If possible, set the switch to use one of the versions %s' % (msg.version, sorted(datapath.supported_ofp_version))
            self._hello_failed(datapath, error_desc)
            return
        datapath.set_version(max(usable_versions))
        self.logger.debug('move onto config mode')
        datapath.set_state(CONFIG_DISPATCHER)
        features_request = datapath.ofproto_parser.OFPFeaturesRequest(datapath)
        datapath.send_msg(features_request)

    @set_ev_handler(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        self.logger.debug('switch features ev %s', msg)
        datapath.id = msg.datapath_id
        if datapath.ofproto.OFP_VERSION < 4:
            datapath.ports = msg.ports
        else:
            datapath.ports = {}
        if datapath.ofproto.OFP_VERSION < 4:
            self.logger.debug('move onto main mode')
            ev.msg.datapath.set_state(MAIN_DISPATCHER)
        else:
            port_desc = datapath.ofproto_parser.OFPPortDescStatsRequest(datapath, 0)
            datapath.send_msg(port_desc)

    @set_ev_handler(ofp_event.EventOFPPortDescStatsReply, CONFIG_DISPATCHER)
    def multipart_reply_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for port in msg.body:
                datapath.ports[port.port_no] = port
        if msg.flags & datapath.ofproto.OFPMPF_REPLY_MORE:
            return
        self.logger.debug('move onto main mode')
        ev.msg.datapath.set_state(MAIN_DISPATCHER)

    @set_ev_handler(ofp_event.EventOFPEchoRequest, [HANDSHAKE_DISPATCHER, CONFIG_DISPATCHER, MAIN_DISPATCHER])
    def echo_request_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        echo_reply = datapath.ofproto_parser.OFPEchoReply(datapath)
        echo_reply.xid = msg.xid
        echo_reply.data = msg.data
        datapath.send_msg(echo_reply)

    @set_ev_handler(ofp_event.EventOFPEchoReply, [HANDSHAKE_DISPATCHER, CONFIG_DISPATCHER, MAIN_DISPATCHER])
    def echo_reply_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        datapath.acknowledge_echo_reply(msg.xid)

    @set_ev_handler(ofp_event.EventOFPPortStatus, MAIN_DISPATCHER)
    def port_status_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        if msg.reason in [ofproto.OFPPR_ADD, ofproto.OFPPR_MODIFY]:
            datapath.ports[msg.desc.port_no] = msg.desc
        elif msg.reason == ofproto.OFPPR_DELETE:
            datapath.ports.pop(msg.desc.port_no, None)
        else:
            return
        self.send_event_to_observers(ofp_event.EventOFPPortStateChange(datapath, msg.reason, msg.desc.port_no), datapath.state)

    @set_ev_handler(ofp_event.EventOFPErrorMsg, [HANDSHAKE_DISPATCHER, CONFIG_DISPATCHER, MAIN_DISPATCHER])
    def error_msg_handler(self, ev):
        msg = ev.msg
        ofp = msg.datapath.ofproto
        self.logger.debug('EventOFPErrorMsg received.\nversion=%s, msg_type=%s, msg_len=%s, xid=%s\n `-- msg_type: %s', hex(msg.version), hex(msg.msg_type), hex(msg.msg_len), hex(msg.xid), ofp.ofp_msg_type_to_str(msg.msg_type))
        if msg.type == ofp.OFPET_EXPERIMENTER:
            self.logger.debug("OFPErrorExperimenterMsg(type=%s, exp_type=%s, experimenter=%s, data=b'%s')", hex(msg.type), hex(msg.exp_type), hex(msg.experimenter), utils.binary_str(msg.data))
        else:
            self.logger.debug("OFPErrorMsg(type=%s, code=%s, data=b'%s')\n |-- type: %s\n |-- code: %s", hex(msg.type), hex(msg.code), utils.binary_str(msg.data), ofp.ofp_error_type_to_str(msg.type), ofp.ofp_error_code_to_str(msg.type, msg.code))
        if msg.type == ofp.OFPET_HELLO_FAILED:
            self.logger.debug(' `-- data: %s', msg.data.decode('ascii'))
        elif len(msg.data) >= ofp.OFP_HEADER_SIZE:
            version, msg_type, msg_len, xid = ofproto_parser.header(msg.data)
            self.logger.debug(' `-- data: version=%s, msg_type=%s, msg_len=%s, xid=%s\n     `-- msg_type: %s', hex(version), hex(msg_type), hex(msg_len), hex(xid), ofp.ofp_msg_type_to_str(msg_type))
        else:
            self.logger.warning('The data field sent from the switch is too short: len(msg.data) < OFP_HEADER_SIZE\nThe OpenFlow Spec says that the data field should contain at least 64 bytes of the failed request.\nPlease check the settings or implementation of your switch.')