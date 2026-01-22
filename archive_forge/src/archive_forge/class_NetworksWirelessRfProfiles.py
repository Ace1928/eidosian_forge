from __future__ import (absolute_import, division, print_function)
from ansible.plugins.action import ActionBase
from ansible.errors import AnsibleActionFail
from ansible_collections.cisco.meraki.plugins.plugin_utils.meraki import (
from ansible_collections.cisco.meraki.plugins.plugin_utils.exceptions import (
class NetworksWirelessRfProfiles(object):

    def __init__(self, params, meraki):
        self.meraki = meraki
        self.new_object = dict(name=params.get('name'), clientBalancingEnabled=params.get('clientBalancingEnabled'), minBitrateType=params.get('minBitrateType'), bandSelectionType=params.get('bandSelectionType'), apBandSettings=params.get('apBandSettings'), twoFourGhzSettings=params.get('twoFourGhzSettings'), fiveGhzSettings=params.get('fiveGhzSettings'), sixGhzSettings=params.get('sixGhzSettings'), transmission=params.get('transmission'), perSsidSettings=params.get('perSsidSettings'), networkId=params.get('networkId'), rfProfileId=params.get('rfProfileId'))

    def get_all_params(self, name=None, id=None):
        new_object_params = {}
        if self.new_object.get('includeTemplateProfiles') is not None or self.new_object.get('include_template_profiles') is not None:
            new_object_params['includeTemplateProfiles'] = self.new_object.get('includeTemplateProfiles') or self.new_object.get('include_template_profiles')
        if self.new_object.get('networkId') is not None or self.new_object.get('network_id') is not None:
            new_object_params['networkId'] = self.new_object.get('networkId') or self.new_object.get('network_id')
        return new_object_params

    def get_params_by_id(self, name=None, id=None):
        new_object_params = {}
        if self.new_object.get('networkId') is not None or self.new_object.get('network_id') is not None:
            new_object_params['networkId'] = self.new_object.get('networkId') or self.new_object.get('network_id')
        if self.new_object.get('rfProfileId') is not None or self.new_object.get('rf_profile_id') is not None:
            new_object_params['rfProfileId'] = self.new_object.get('rfProfileId') or self.new_object.get('rf_profile_id')
        return new_object_params

    def create_params(self):
        new_object_params = {}
        if self.new_object.get('name') is not None or self.new_object.get('name') is not None:
            new_object_params['name'] = self.new_object.get('name') or self.new_object.get('name')
        if self.new_object.get('clientBalancingEnabled') is not None or self.new_object.get('client_balancing_enabled') is not None:
            new_object_params['clientBalancingEnabled'] = self.new_object.get('clientBalancingEnabled')
        if self.new_object.get('minBitrateType') is not None or self.new_object.get('min_bitrate_type') is not None:
            new_object_params['minBitrateType'] = self.new_object.get('minBitrateType') or self.new_object.get('min_bitrate_type')
        if self.new_object.get('bandSelectionType') is not None or self.new_object.get('band_selection_type') is not None:
            new_object_params['bandSelectionType'] = self.new_object.get('bandSelectionType') or self.new_object.get('band_selection_type')
        if self.new_object.get('apBandSettings') is not None or self.new_object.get('ap_band_settings') is not None:
            new_object_params['apBandSettings'] = self.new_object.get('apBandSettings') or self.new_object.get('ap_band_settings')
        if self.new_object.get('twoFourGhzSettings') is not None or self.new_object.get('two_four_ghz_settings') is not None:
            new_object_params['twoFourGhzSettings'] = self.new_object.get('twoFourGhzSettings') or self.new_object.get('two_four_ghz_settings')
        if self.new_object.get('fiveGhzSettings') is not None or self.new_object.get('five_ghz_settings') is not None:
            new_object_params['fiveGhzSettings'] = self.new_object.get('fiveGhzSettings') or self.new_object.get('five_ghz_settings')
        if self.new_object.get('sixGhzSettings') is not None or self.new_object.get('six_ghz_settings') is not None:
            new_object_params['sixGhzSettings'] = self.new_object.get('sixGhzSettings') or self.new_object.get('six_ghz_settings')
        if self.new_object.get('transmission') is not None or self.new_object.get('transmission') is not None:
            new_object_params['transmission'] = self.new_object.get('transmission') or self.new_object.get('transmission')
        if self.new_object.get('perSsidSettings') is not None or self.new_object.get('per_ssid_settings') is not None:
            new_object_params['perSsidSettings'] = self.new_object.get('perSsidSettings') or self.new_object.get('per_ssid_settings')
        if self.new_object.get('networkId') is not None or self.new_object.get('network_id') is not None:
            new_object_params['networkId'] = self.new_object.get('networkId') or self.new_object.get('network_id')
        return new_object_params

    def delete_by_id_params(self):
        new_object_params = {}
        if self.new_object.get('networkId') is not None or self.new_object.get('network_id') is not None:
            new_object_params['networkId'] = self.new_object.get('networkId') or self.new_object.get('network_id')
        if self.new_object.get('rfProfileId') is not None or self.new_object.get('rf_profile_id') is not None:
            new_object_params['rfProfileId'] = self.new_object.get('rfProfileId') or self.new_object.get('rf_profile_id')
        return new_object_params

    def update_by_id_params(self):
        new_object_params = {}
        if self.new_object.get('name') is not None or self.new_object.get('name') is not None:
            new_object_params['name'] = self.new_object.get('name') or self.new_object.get('name')
        if self.new_object.get('clientBalancingEnabled') is not None or self.new_object.get('client_balancing_enabled') is not None:
            new_object_params['clientBalancingEnabled'] = self.new_object.get('clientBalancingEnabled')
        if self.new_object.get('minBitrateType') is not None or self.new_object.get('min_bitrate_type') is not None:
            new_object_params['minBitrateType'] = self.new_object.get('minBitrateType') or self.new_object.get('min_bitrate_type')
        if self.new_object.get('bandSelectionType') is not None or self.new_object.get('band_selection_type') is not None:
            new_object_params['bandSelectionType'] = self.new_object.get('bandSelectionType') or self.new_object.get('band_selection_type')
        if self.new_object.get('apBandSettings') is not None or self.new_object.get('ap_band_settings') is not None:
            new_object_params['apBandSettings'] = self.new_object.get('apBandSettings') or self.new_object.get('ap_band_settings')
        if self.new_object.get('twoFourGhzSettings') is not None or self.new_object.get('two_four_ghz_settings') is not None:
            new_object_params['twoFourGhzSettings'] = self.new_object.get('twoFourGhzSettings') or self.new_object.get('two_four_ghz_settings')
        if self.new_object.get('fiveGhzSettings') is not None or self.new_object.get('five_ghz_settings') is not None:
            new_object_params['fiveGhzSettings'] = self.new_object.get('fiveGhzSettings') or self.new_object.get('five_ghz_settings')
        if self.new_object.get('sixGhzSettings') is not None or self.new_object.get('six_ghz_settings') is not None:
            new_object_params['sixGhzSettings'] = self.new_object.get('sixGhzSettings') or self.new_object.get('six_ghz_settings')
        if self.new_object.get('transmission') is not None or self.new_object.get('transmission') is not None:
            new_object_params['transmission'] = self.new_object.get('transmission') or self.new_object.get('transmission')
        if self.new_object.get('perSsidSettings') is not None or self.new_object.get('per_ssid_settings') is not None:
            new_object_params['perSsidSettings'] = self.new_object.get('perSsidSettings') or self.new_object.get('per_ssid_settings')
        if self.new_object.get('networkId') is not None or self.new_object.get('network_id') is not None:
            new_object_params['networkId'] = self.new_object.get('networkId') or self.new_object.get('network_id')
        if self.new_object.get('rfProfileId') is not None or self.new_object.get('rf_profile_id') is not None:
            new_object_params['rfProfileId'] = self.new_object.get('rfProfileId') or self.new_object.get('rf_profile_id')
        return new_object_params

    def get_object_by_name(self, name):
        result = None
        try:
            items = self.meraki.exec_meraki(family='wireless', function='getNetworkWirelessRfProfiles', params=self.get_all_params(name=name))
            if isinstance(items, dict):
                if 'response' in items:
                    items = items.get('response')
            result = get_dict_result(items, 'name', name)
            if result is None:
                result = items
        except Exception as e:
            print('Error: ', e)
            result = None
        return result

    def get_object_by_id(self, id):
        result = None
        try:
            items = self.meraki.exec_meraki(family='wireless', function='getNetworkWirelessRfProfile', params=self.get_params_by_id())
            if isinstance(items, dict):
                if 'response' in items:
                    items = items.get('response')
            result = get_dict_result(items, 'rfProfileId', id)
        except Exception as e:
            print('Error: ', e)
            result = None
        return result

    def exists(self):
        id_exists = False
        name_exists = False
        prev_obj = None
        o_id = self.new_object.get('id')
        o_id = o_id or self.new_object.get('rf_profile_id') or self.new_object.get('rfProfileId')
        name = self.new_object.get('name')
        if o_id:
            prev_obj = self.get_object_by_id(o_id)
            id_exists = prev_obj is not None and isinstance(prev_obj, dict)
        if not id_exists and name:
            prev_obj = self.get_object_by_name(name)
            name_exists = prev_obj is not None and isinstance(prev_obj, dict)
        if name_exists:
            _id = prev_obj.get('id')
            _id = _id or prev_obj.get('rfProfileId')
            if id_exists and name_exists and (o_id != _id):
                raise InconsistentParameters("The 'id' and 'name' params don't refer to the same object")
            if _id:
                self.new_object.update(dict(id=_id))
                self.new_object.update(dict(rfProfileId=_id))
            if _id:
                prev_obj = self.get_object_by_id(_id)
        it_exists = prev_obj is not None and isinstance(prev_obj, dict)
        return (it_exists, prev_obj)

    def requires_update(self, current_obj):
        requested_obj = self.new_object
        obj_params = [('name', 'name'), ('clientBalancingEnabled', 'clientBalancingEnabled'), ('minBitrateType', 'minBitrateType'), ('bandSelectionType', 'bandSelectionType'), ('apBandSettings', 'apBandSettings'), ('twoFourGhzSettings', 'twoFourGhzSettings'), ('fiveGhzSettings', 'fiveGhzSettings'), ('sixGhzSettings', 'sixGhzSettings'), ('transmission', 'transmission'), ('perSsidSettings', 'perSsidSettings'), ('networkId', 'networkId'), ('rfProfileId', 'rfProfileId')]
        return any((not meraki_compare_equality(current_obj.get(meraki_param), requested_obj.get(ansible_param)) for meraki_param, ansible_param in obj_params))

    def create(self):
        result = self.meraki.exec_meraki(family='wireless', function='createNetworkWirelessRfProfile', params=self.create_params(), op_modifies=True)
        return result

    def update(self):
        id = self.new_object.get('id')
        id = id or self.new_object.get('rfProfileId')
        name = self.new_object.get('name')
        result = None
        if not id:
            prev_obj_name = self.get_object_by_name(name)
            id_ = None
            if prev_obj_name:
                id_ = prev_obj_name.get('id')
                id_ = id_ or prev_obj_name.get('rfProfileId')
            if id_:
                self.new_object.update(dict(rfProfileId=id_))
        result = self.meraki.exec_meraki(family='wireless', function='updateNetworkWirelessRfProfile', params=self.update_by_id_params(), op_modifies=True)
        return result

    def delete(self):
        id = self.new_object.get('id')
        id = id or self.new_object.get('rfProfileId')
        name = self.new_object.get('name')
        result = None
        if not id:
            prev_obj_name = self.get_object_by_name(name)
            id_ = None
            if prev_obj_name:
                id_ = prev_obj_name.get('id')
                id_ = id_ or prev_obj_name.get('rfProfileId')
            if id_:
                self.new_object.update(dict(rfProfileId=id_))
        result = self.meraki.exec_meraki(family='wireless', function='deleteNetworkWirelessRfProfile', params=self.delete_by_id_params())
        return result