from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, ce_argument_spec
class InfoCenterDebug(object):
    """ Manages info center debug configuration """

    def __init__(self, **kwargs):
        """ Init function """
        argument_spec = kwargs['argument_spec']
        self.spec = argument_spec
        self.module = AnsibleModule(argument_spec=self.spec, supports_check_mode=True)
        self.state = self.module.params['state']
        self.debug_time_stamp = self.module.params['debug_time_stamp'] or None
        self.module_name = self.module.params['module_name'] or None
        self.channel_id = self.module.params['channel_id'] or None
        self.debug_enable = self.module.params['debug_enable']
        self.debug_level = self.module.params['debug_level'] or None
        self.cur_global_cfg = dict()
        self.cur_source_cfg = dict()
        self.changed = False
        self.updates_cmd = list()
        self.results = dict()
        self.proposed = dict()
        self.existing = dict()
        self.end_state = dict()

    def check_global_args(self):
        """ Check global args """
        need_cfg = False
        find_flag = False
        self.cur_global_cfg['global_cfg'] = []
        if self.debug_time_stamp:
            conf_str = CE_GET_DEBUG_GLOBAL_HEADER
            conf_str += '<debugTimeStamp></debugTimeStamp>'
            conf_str += CE_GET_DEBUG_GLOBAL_TAIL
            xml_str = get_nc_config(self.module, conf_str)
            if '<data/>' in xml_str:
                find_flag = False
            else:
                xml_str = xml_str.replace('\r', '').replace('\n', '').replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '').replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
                root = ElementTree.fromstring(xml_str)
                global_cfg = root.findall('syslog/globalParam')
                if global_cfg:
                    for tmp in global_cfg:
                        tmp_dict = dict()
                        for site in tmp:
                            if site.tag in ['debugTimeStamp']:
                                tmp_dict[site.tag] = site.text
                        self.cur_global_cfg['global_cfg'].append(tmp_dict)
                if self.cur_global_cfg['global_cfg']:
                    for tmp in self.cur_global_cfg['global_cfg']:
                        find_flag = True
                        if tmp.get('debugTimeStamp').lower() != self.debug_time_stamp:
                            find_flag = False
                        if find_flag:
                            break
                else:
                    find_flag = False
            if self.state == 'present':
                need_cfg = bool(not find_flag)
            else:
                need_cfg = bool(find_flag)
        self.cur_global_cfg['need_cfg'] = need_cfg

    def check_source_args(self):
        """ Check source args """
        need_cfg = False
        find_flag = False
        self.cur_source_cfg['source_cfg'] = []
        if self.module_name:
            if len(self.module_name) < 1 or len(self.module_name) > 31:
                self.module.fail_json(msg='Error: The module_name is out of [1 - 31].')
            if not self.channel_id:
                self.module.fail_json(msg='Error: Please input channel_id at the same time.')
            if self.channel_id:
                if self.channel_id.isdigit():
                    if int(self.channel_id) < 0 or int(self.channel_id) > 9:
                        self.module.fail_json(msg='Error: The value of channel_id is out of [0 - 9].')
                else:
                    self.module.fail_json(msg='Error: The channel_id is not digit.')
            conf_str = CE_GET_DEBUG_SOURCE_HEADER
            if self.module_name != 'default':
                conf_str += '<moduleName>%s</moduleName>' % self.module_name.upper()
            else:
                conf_str += '<moduleName>default</moduleName>'
            if self.channel_id:
                conf_str += '<icChannelId></icChannelId>'
            if self.debug_enable != 'no_use':
                conf_str += '<dbgEnFlg></dbgEnFlg>'
            if self.debug_level:
                conf_str += '<dbgEnLevel></dbgEnLevel>'
            conf_str += CE_GET_DEBUG_SOURCE_TAIL
            xml_str = get_nc_config(self.module, conf_str)
            if '<data/>' in xml_str:
                find_flag = False
            else:
                xml_str = xml_str.replace('\r', '').replace('\n', '').replace('xmlns="urn:ietf:params:xml:ns:netconf:base:1.0"', '').replace('xmlns="http://www.huawei.com/netconf/vrp"', '')
                root = ElementTree.fromstring(xml_str)
                source_cfg = root.findall('syslog/icSources/icSource')
                if source_cfg:
                    for tmp in source_cfg:
                        tmp_dict = dict()
                        for site in tmp:
                            if site.tag in ['moduleName', 'icChannelId', 'dbgEnFlg', 'dbgEnLevel']:
                                tmp_dict[site.tag] = site.text
                        self.cur_source_cfg['source_cfg'].append(tmp_dict)
                if self.cur_source_cfg['source_cfg']:
                    for tmp in self.cur_source_cfg['source_cfg']:
                        find_flag = True
                        if self.module_name and tmp.get('moduleName').lower() != self.module_name.lower():
                            find_flag = False
                        if self.channel_id and tmp.get('icChannelId') != self.channel_id:
                            find_flag = False
                        if self.debug_enable != 'no_use' and tmp.get('dbgEnFlg') != self.debug_enable:
                            find_flag = False
                        if self.debug_level and tmp.get('dbgEnLevel') != self.debug_level:
                            find_flag = False
                        if find_flag:
                            break
                else:
                    find_flag = False
            if self.state == 'present':
                need_cfg = bool(not find_flag)
            else:
                need_cfg = bool(find_flag)
        self.cur_source_cfg['need_cfg'] = need_cfg

    def get_proposed(self):
        """ Get proposed """
        self.proposed['state'] = self.state
        if self.debug_time_stamp:
            self.proposed['debug_time_stamp'] = self.debug_time_stamp
        if self.module_name:
            self.proposed['module_name'] = self.module_name
        if self.channel_id:
            self.proposed['channel_id'] = self.channel_id
        if self.debug_enable != 'no_use':
            self.proposed['debug_enable'] = self.debug_enable
        if self.debug_level:
            self.proposed['debug_level'] = self.debug_level

    def get_existing(self):
        """ Get existing """
        if self.cur_global_cfg['global_cfg']:
            self.existing['global_cfg'] = self.cur_global_cfg['global_cfg']
        if self.cur_source_cfg['source_cfg']:
            self.existing['source_cfg'] = self.cur_source_cfg['source_cfg']

    def get_end_state(self):
        """ Get end state """
        self.check_global_args()
        if self.cur_global_cfg['global_cfg']:
            self.end_state['global_cfg'] = self.cur_global_cfg['global_cfg']
        self.check_source_args()
        if self.cur_source_cfg['source_cfg']:
            self.end_state['source_cfg'] = self.cur_source_cfg['source_cfg']

    def merge_debug_global(self):
        """ Merge debug global """
        conf_str = CE_MERGE_DEBUG_GLOBAL_HEADER
        if self.debug_time_stamp:
            conf_str += '<debugTimeStamp>%s</debugTimeStamp>' % self.debug_time_stamp.upper()
        conf_str += CE_MERGE_DEBUG_GLOBAL_TAIL
        recv_xml = set_nc_config(self.module, conf_str)
        if '<ok/>' not in recv_xml:
            self.module.fail_json(msg='Error: Merge debug global failed.')
        if self.debug_time_stamp:
            cmd = 'info-center timestamp debugging ' + TIME_STAMP_DICT.get(self.debug_time_stamp)
            self.updates_cmd.append(cmd)
        self.changed = True

    def delete_debug_global(self):
        """ Delete debug global """
        conf_str = CE_MERGE_DEBUG_GLOBAL_HEADER
        if self.debug_time_stamp:
            conf_str += '<debugTimeStamp>DATE_MILLISECOND</debugTimeStamp>'
        conf_str += CE_MERGE_DEBUG_GLOBAL_TAIL
        recv_xml = set_nc_config(self.module, conf_str)
        if '<ok/>' not in recv_xml:
            self.module.fail_json(msg='Error: delete debug global failed.')
        if self.debug_time_stamp:
            cmd = 'undo info-center timestamp debugging'
            self.updates_cmd.append(cmd)
        self.changed = True

    def merge_debug_source(self):
        """ Merge debug source """
        conf_str = CE_MERGE_DEBUG_SOURCE_HEADER
        if self.module_name:
            conf_str += '<moduleName>%s</moduleName>' % self.module_name
        if self.channel_id:
            conf_str += '<icChannelId>%s</icChannelId>' % self.channel_id
        if self.debug_enable != 'no_use':
            conf_str += '<dbgEnFlg>%s</dbgEnFlg>' % self.debug_enable
        if self.debug_level:
            conf_str += '<dbgEnLevel>%s</dbgEnLevel>' % self.debug_level
        conf_str += CE_MERGE_DEBUG_SOURCE_TAIL
        recv_xml = set_nc_config(self.module, conf_str)
        if '<ok/>' not in recv_xml:
            self.module.fail_json(msg='Error: Merge debug source failed.')
        cmd = 'info-center source'
        if self.module_name:
            cmd += ' %s' % self.module_name
        if self.channel_id:
            cmd += ' channel %s' % self.channel_id
        if self.debug_enable != 'no_use':
            if self.debug_enable == 'true':
                cmd += ' debug state on'
            else:
                cmd += ' debug state off'
        if self.debug_level:
            cmd += ' level %s' % self.debug_level
        self.updates_cmd.append(cmd)
        self.changed = True

    def delete_debug_source(self):
        """ Delete debug source """
        if self.debug_enable == 'no_use' and (not self.debug_level):
            conf_str = CE_DELETE_DEBUG_SOURCE_HEADER
            if self.module_name:
                conf_str += '<moduleName>%s</moduleName>' % self.module_name
            if self.channel_id:
                conf_str += '<icChannelId>%s</icChannelId>' % self.channel_id
            conf_str += CE_DELETE_DEBUG_SOURCE_TAIL
        else:
            conf_str = CE_MERGE_DEBUG_SOURCE_HEADER
            if self.module_name:
                conf_str += '<moduleName>%s</moduleName>' % self.module_name
            if self.channel_id:
                conf_str += '<icChannelId>%s</icChannelId>' % self.channel_id
            if self.debug_enable != 'no_use':
                conf_str += '<dbgEnFlg>%s</dbgEnFlg>' % CHANNEL_DEFAULT_DBG_STATE.get(self.channel_id)
            if self.debug_level:
                conf_str += '<dbgEnLevel>%s</dbgEnLevel>' % CHANNEL_DEFAULT_DBG_LEVEL.get(self.channel_id)
            conf_str += CE_MERGE_DEBUG_SOURCE_TAIL
        recv_xml = set_nc_config(self.module, conf_str)
        if '<ok/>' not in recv_xml:
            self.module.fail_json(msg='Error: Delete debug source failed.')
        cmd = 'undo info-center source'
        if self.module_name:
            cmd += ' %s' % self.module_name
        if self.channel_id:
            cmd += ' channel %s' % self.channel_id
        if self.debug_enable != 'no_use':
            cmd += ' debug state'
        if self.debug_level:
            cmd += ' level'
        self.updates_cmd.append(cmd)
        self.changed = True

    def work(self):
        """ work function """
        self.check_global_args()
        self.check_source_args()
        self.get_proposed()
        self.get_existing()
        if self.state == 'present':
            if self.cur_global_cfg['need_cfg']:
                self.merge_debug_global()
            if self.cur_source_cfg['need_cfg']:
                self.merge_debug_source()
        else:
            if self.cur_global_cfg['need_cfg']:
                self.delete_debug_global()
            if self.cur_source_cfg['need_cfg']:
                self.delete_debug_source()
        self.get_end_state()
        self.results['changed'] = self.changed
        self.results['proposed'] = self.proposed
        self.results['existing'] = self.existing
        self.results['end_state'] = self.end_state
        self.results['updates'] = self.updates_cmd
        self.module.exit_json(**self.results)