from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
class InfoCenterLog(object):
    """
    Manages information center log configuration
    """

    def __init__(self, argument_spec):
        self.spec = argument_spec
        self.module = None
        self.init_module()
        self.log_time_stamp = self.module.params['log_time_stamp']
        self.log_buff_enable = self.module.params['log_buff_enable']
        self.log_buff_size = self.module.params['log_buff_size']
        self.module_name = self.module.params['module_name']
        self.channel_id = self.module.params['channel_id']
        self.log_enable = self.module.params['log_enable']
        self.log_level = self.module.params['log_level']
        self.state = self.module.params['state']
        self.log_dict = dict()
        self.changed = False
        self.updates_cmd = list()
        self.commands = list()
        self.results = dict()
        self.proposed = dict()
        self.existing = dict()
        self.end_state = dict()

    def init_module(self):
        """init module"""
        self.module = AnsibleModule(argument_spec=self.spec, supports_check_mode=True)

    def check_response(self, xml_str, xml_name):
        """Check if response message is already succeed"""
        if '<ok/>' not in xml_str:
            self.module.fail_json(msg='Error: %s failed.' % xml_name)

    def get_log_dict(self):
        """ log config dict"""
        log_dict = dict()
        if self.module_name:
            if self.module_name.lower() == 'default':
                conf_str = CE_NC_GET_LOG % (self.module_name.lower(), self.channel_id)
            else:
                conf_str = CE_NC_GET_LOG % (self.module_name.upper(), self.channel_id)
        else:
            conf_str = CE_NC_GET_LOG_GLOBAL
        xml_str = get_nc_config(self.module, conf_str)
        if '<data/>' in xml_str:
            return log_dict
        xml_str = xml_str.replace('\r', '').replace('\n', '').replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '').replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
        root = ElementTree.fromstring(xml_str)
        glb = root.find('syslog/globalParam')
        if glb:
            for attr in glb:
                if attr.tag in ['bufferSize', 'logTimeStamp', 'icLogBuffEn']:
                    log_dict[attr.tag] = attr.text
        log_dict['source'] = dict()
        src = root.find('syslog/icSources/icSource')
        if src:
            for attr in src:
                if attr.tag in ['moduleName', 'icChannelId', 'icChannelName', 'logEnFlg', 'logEnLevel']:
                    log_dict['source'][attr.tag] = attr.text
        return log_dict

    def config_log_global(self):
        """config log global param"""
        xml_str = '<globalParam operation="merge">'
        if self.log_time_stamp:
            if self.state == 'present' and self.log_time_stamp.upper() != self.log_dict.get('logTimeStamp'):
                xml_str += '<logTimeStamp>%s</logTimeStamp>' % self.log_time_stamp.upper()
                self.updates_cmd.append('info-center timestamp log %s' % TIME_STAMP_DICT.get(self.log_time_stamp))
            elif self.state == 'absent' and self.log_time_stamp.upper() == self.log_dict.get('logTimeStamp'):
                xml_str += '<logTimeStamp>DATE_SECOND</logTimeStamp>'
                self.updates_cmd.append('undo info-center timestamp log')
            else:
                pass
        if self.log_buff_enable != 'no_use':
            if self.log_dict.get('icLogBuffEn') != self.log_buff_enable:
                xml_str += '<icLogBuffEn>%s</icLogBuffEn>' % self.log_buff_enable
                if self.log_buff_enable == 'true':
                    self.updates_cmd.append('info-center logbuffer')
                else:
                    self.updates_cmd.append('undo info-center logbuffer')
        if self.log_buff_size:
            if self.state == 'present' and self.log_dict.get('bufferSize') != self.log_buff_size:
                xml_str += '<bufferSize>%s</bufferSize>' % self.log_buff_size
                self.updates_cmd.append('info-center logbuffer size %s' % self.log_buff_size)
            elif self.state == 'absent' and self.log_dict.get('bufferSize') == self.log_buff_size:
                xml_str += '<bufferSize>512</bufferSize>'
                self.updates_cmd.append('undo info-center logbuffer size')
        if xml_str == '<globalParam operation="merge">':
            return ''
        else:
            xml_str += '</globalParam>'
            return xml_str

    def config_log_soruce(self):
        """config info-center sources"""
        xml_str = ''
        if not self.module_name or not self.channel_id:
            return xml_str
        source = self.log_dict['source']
        if self.state == 'present':
            xml_str = '<icSources><icSource operation="merge">'
            cmd = 'info-center source %s channel %s log' % (self.module_name, self.channel_id)
        else:
            if not source or self.module_name != source.get('moduleName').lower() or self.channel_id != source.get('icChannelId'):
                return ''
            if self.log_enable == 'no_use' and (not self.log_level):
                xml_str = '<icSources><icSource operation="delete">'
            else:
                xml_str = '<icSources><icSource operation="merge">'
            cmd = 'undo info-center source %s channel %s log' % (self.module_name, self.channel_id)
        xml_str += '<moduleName>%s</moduleName><icChannelId>%s</icChannelId>' % (self.module_name, self.channel_id)
        if self.log_enable != 'no_use':
            if self.state == 'present' and (not source or self.log_enable != source.get('logEnFlg')):
                xml_str += '<logEnFlg>%s</logEnFlg>' % self.log_enable
                if self.log_enable == 'true':
                    cmd += ' state on'
                else:
                    cmd += ' state off'
            elif self.state == 'absent' and source and (self.log_level == source.get('logEnLevel')):
                xml_str += '<logEnFlg>%s</logEnFlg>' % CHANNEL_DEFAULT_LOG_STATE.get(self.channel_id)
                cmd += ' state'
        if self.log_level:
            if self.state == 'present' and (not source or self.log_level != source.get('logEnLevel')):
                xml_str += '<logEnLevel>%s</logEnLevel>' % self.log_level
                cmd += ' level %s' % self.log_level
            elif self.state == 'absent' and source and (self.log_level == source.get('logEnLevel')):
                xml_str += '<logEnLevel>%s</logEnLevel>' % CHANNEL_DEFAULT_LOG_LEVEL.get(self.channel_id)
                cmd += ' level'
        if xml_str.endswith('</icChannelId>'):
            if self.log_enable == 'no_use' and (not self.log_level) and (self.state == 'absent'):
                xml_str += '</icSource></icSources>'
                self.updates_cmd.append(cmd)
                return xml_str
            else:
                return ''
        else:
            xml_str += '</icSource></icSources>'
            self.updates_cmd.append(cmd)
            return xml_str

    def netconf_load_config(self, xml_str):
        """load log config by netconf"""
        if not xml_str:
            return
        xml_cfg = '\n            <config>\n            <syslog xmlns="http://www.huawei.com/netconf/vrp" content-version="1.0" format-version="1.0">\n            %s\n            </syslog>\n            </config>' % xml_str
        recv_xml = set_nc_config(self.module, xml_cfg)
        self.check_response(recv_xml, 'SET_LOG')
        self.changed = True

    def check_params(self):
        """Check all input params"""
        if self.log_buff_size:
            if not self.log_buff_size.isdigit():
                self.module.fail_json(msg='Error: log_buff_size is not digit.')
            if int(self.log_buff_size) < 0 or int(self.log_buff_size) > 10240:
                self.module.fail_json(msg='Error: log_buff_size is not ranges from 0 to 10240.')
        if self.channel_id:
            if not self.channel_id.isdigit():
                self.module.fail_json(msg='Error: channel_id is not digit.')
            if int(self.channel_id) < 0 or int(self.channel_id) > 9:
                self.module.fail_json(msg='Error: channel_id is not ranges from 0 to 9.')
        if bool(self.module_name) != bool(self.channel_id):
            self.module.fail_json(msg='Error: module_name and channel_id must be set at the same time.')

    def get_proposed(self):
        """get proposed info"""
        if self.log_time_stamp:
            self.proposed['log_time_stamp'] = self.log_time_stamp
        if self.log_buff_enable != 'no_use':
            self.proposed['log_buff_enable'] = self.log_buff_enable
        if self.log_buff_size:
            self.proposed['log_buff_size'] = self.log_buff_size
        if self.module_name:
            self.proposed['module_name'] = self.module_name
        if self.channel_id:
            self.proposed['channel_id'] = self.channel_id
        if self.log_enable != 'no_use':
            self.proposed['log_enable'] = self.log_enable
        if self.log_level:
            self.proposed['log_level'] = self.log_level
        self.proposed['state'] = self.state

    def get_existing(self):
        """get existing info"""
        if not self.log_dict:
            return
        if self.log_time_stamp:
            self.existing['log_time_stamp'] = self.log_dict.get('logTimeStamp').lower()
        if self.log_buff_enable != 'no_use':
            self.existing['log_buff_enable'] = self.log_dict.get('icLogBuffEn')
        if self.log_buff_size:
            self.existing['log_buff_size'] = self.log_dict.get('bufferSize')
        if self.module_name:
            self.existing['source'] = self.log_dict.get('source')

    def get_end_state(self):
        """get end state info"""
        log_dict = self.get_log_dict()
        if not log_dict:
            return
        if self.log_time_stamp:
            self.end_state['log_time_stamp'] = log_dict.get('logTimeStamp').lower()
        if self.log_buff_enable != 'no_use':
            self.end_state['log_buff_enable'] = log_dict.get('icLogBuffEn')
        if self.log_buff_size:
            self.end_state['log_buff_size'] = log_dict.get('bufferSize')
        if self.module_name:
            self.end_state['source'] = log_dict.get('source')

    def work(self):
        """worker"""
        self.check_params()
        self.log_dict = self.get_log_dict()
        self.get_existing()
        self.get_proposed()
        xml_str = ''
        if self.log_time_stamp or self.log_buff_enable != 'no_use' or self.log_buff_size:
            xml_str += self.config_log_global()
        if self.module_name:
            xml_str += self.config_log_soruce()
        if xml_str:
            self.netconf_load_config(xml_str)
            self.changed = True
        self.get_end_state()
        self.results['changed'] = self.changed
        self.results['proposed'] = self.proposed
        self.results['existing'] = self.existing
        self.results['end_state'] = self.end_state
        if self.changed:
            self.results['updates'] = self.updates_cmd
        else:
            self.results['updates'] = list()
        self.module.exit_json(**self.results)