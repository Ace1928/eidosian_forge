from __future__ import absolute_import, division, print_function
import time
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
class V1ModuleParameters(Parameters):

    @property
    def template(self):
        if self._values['template'] is None:
            return None
        template_map = {'ActiveSync v1.0 v2.0 (http)': 'POLICY_TEMPLATE_ACTIVESYNC_V1_0_V2_0_HTTP', 'ActiveSync v1.0 v2.0 (https)': 'POLICY_TEMPLATE_ACTIVESYNC_V1_0_V2_0_HTTPS', 'LotusDomino 6.5 (http)': 'POLICY_TEMPLATE_LOTUSDOMINO_6_5_HTTP', 'LotusDomino 6.5 (https)': 'POLICY_TEMPLATE_LOTUSDOMINO_6_5_HTTPS', 'OWA Exchange 2003 (http)': 'POLICY_TEMPLATE_OWA_EXCHANGE_2003_HTTP', 'OWA Exchange 2003 (https)': 'POLICY_TEMPLATE_OWA_EXCHANGE_2003_HTTPS', 'OWA Exchange 2003 with ActiveSync (http)': 'POLICY_TEMPLATE_OWA_EXCHANGE_2003_WITH_ACTIVESYNC_HTTP', 'OWA Exchange 2003 with ActiveSync (https)': 'POLICY_TEMPLATE_OWA_EXCHANGE_2003_WITH_ACTIVESYNC_HTTPS', 'OWA Exchange 2007 (http)': 'POLICY_TEMPLATE_OWA_EXCHANGE_2007_HTTP', 'OWA Exchange 2007 (https)': 'POLICY_TEMPLATE_OWA_EXCHANGE_2007_HTTPS', 'OWA Exchange 2007 with ActiveSync (http)': 'POLICY_TEMPLATE_OWA_EXCHANGE_2007_WITH_ACTIVESYNC_HTTP', 'OWA Exchange 2007 with ActiveSync (https)': 'POLICY_TEMPLATE_OWA_EXCHANGE_2007_WITH_ACTIVESYNC_HTTPS', 'OWA Exchange 2010 (http)': 'POLICY_TEMPLATE_OWA_EXCHANGE_2010_HTTP', 'OWA Exchange 2010 (https)': 'POLICY_TEMPLATE_OWA_EXCHANGE_2010_HTTPS', 'Oracle 10g Portal (http)': 'POLICY_TEMPLATE_ORACLE_10G_PORTAL_HTTP', 'Oracle 10g Portal (https)': 'POLICY_TEMPLATE_ORACLE_10G_PORTAL_HTTPS', 'Oracle Applications 11i (http)': 'POLICY_TEMPLATE_ORACLE_APPLICATIONS_11I_HTTP', 'Oracle Applications 11i (https)': 'POLICY_TEMPLATE_ORACLE_APPLICATIONS_11I_HTTPS', 'PeopleSoft Portal 9 (http)': 'POLICY_TEMPLATE_PEOPLESOFT_PORTAL_9_HTTP', 'PeopleSoft Portal 9 (https)': 'POLICY_TEMPLATE_PEOPLESOFT_PORTAL_9_HTTPS', 'Rapid Deployment Policy': 'POLICY_TEMPLATE_RAPID_DEPLOYMENT', 'SAP NetWeaver 7 (http)': 'POLICY_TEMPLATE_SAP_NETWEAVER_7_HTTP', 'SAP NetWeaver 7 (https)': 'POLICY_TEMPLATE_SAP_NETWEAVER_7_HTTPS', 'SharePoint 2003 (http)': 'POLICY_TEMPLATE_SHAREPOINT_2003_HTTP', 'SharePoint 2003 (https)': 'POLICY_TEMPLATE_SHAREPOINT_2003_HTTPS', 'SharePoint 2007 (http)': 'POLICY_TEMPLATE_SHAREPOINT_2007_HTTP', 'SharePoint 2007 (https)': 'POLICY_TEMPLATE_SHAREPOINT_2007_HTTPS', 'SharePoint 2010 (http)': 'POLICY_TEMPLATE_SHAREPOINT_2010_HTTP', 'SharePoint 2010 (https)': 'POLICY_TEMPLATE_SHAREPOINT_2010_HTTPS'}
        if self._values['template'] in template_map:
            return template_map[self._values['template']]
        else:
            raise F5ModuleError('The specified template is not valid for this version of BIG-IP.')

    @property
    def apply(self):
        result = flatten_boolean(self._values['apply'])
        if result is None:
            return None
        if result == 'yes':
            return True
        return False