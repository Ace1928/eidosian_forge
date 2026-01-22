from __future__ import absolute_import, division, print_function
import re
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.six import iteritems
class BalancerMember(object):
    """ Apache 2.4 mod_proxy LB balancer member.
    attributes:
        read-only:
            host -> member host (string),
            management_url -> member management url (string),
            protocol -> member protocol (string)
            port -> member port (string),
            path -> member location (string),
            balancer_url -> url of this member's parent balancer (string),
            attributes -> whole member attributes (dictionary)
            module -> ansible module instance (AnsibleModule object).
        writable:
            status -> status of the member (dictionary)
    """

    def __init__(self, management_url, balancer_url, module):
        self.host = regexp_extraction(management_url, str(EXPRESSION), 4)
        self.management_url = str(management_url)
        self.protocol = regexp_extraction(management_url, EXPRESSION, 3)
        self.port = regexp_extraction(management_url, EXPRESSION, 5)
        self.path = regexp_extraction(management_url, EXPRESSION, 6)
        self.balancer_url = str(balancer_url)
        self.module = module

    def get_member_attributes(self):
        """ Returns a dictionary of a balancer member's attributes."""
        balancer_member_page = fetch_url(self.module, self.management_url)
        if balancer_member_page[1]['status'] != 200:
            self.module.fail_json(msg='Could not get balancer_member_page, check for connectivity! ' + balancer_member_page[1])
        else:
            try:
                soup = BeautifulSoup(balancer_member_page[0])
            except TypeError as exc:
                self.module.fail_json(msg='Cannot parse balancer_member_page HTML! ' + str(exc))
            else:
                subsoup = soup.findAll('table')[1].findAll('tr')
                keys = subsoup[0].findAll('th')
                for valuesset in subsoup[1::1]:
                    if re.search(pattern=self.host, string=str(valuesset)):
                        values = valuesset.findAll('td')
                        return dict(((keys[x].string, values[x].string) for x in range(0, len(keys))))

    def get_member_status(self):
        """ Returns a dictionary of a balancer member's status attributes."""
        status_mapping = {'disabled': 'Dis', 'drained': 'Drn', 'hot_standby': 'Stby', 'ignore_errors': 'Ign'}
        actual_status = str(self.attributes['Status'])
        status = dict(((mode, patt in actual_status) for mode, patt in iteritems(status_mapping)))
        return status

    def set_member_status(self, values):
        """ Sets a balancer member's status attributes amongst pre-mapped values."""
        values_mapping = {'disabled': '&w_status_D', 'drained': '&w_status_N', 'hot_standby': '&w_status_H', 'ignore_errors': '&w_status_I'}
        request_body = regexp_extraction(self.management_url, EXPRESSION, 1)
        values_url = ''.join(('{0}={1}'.format(url_param, 1 if values[mode] else 0) for mode, url_param in iteritems(values_mapping)))
        request_body = '{0}{1}'.format(request_body, values_url)
        response = fetch_url(self.module, self.management_url, data=request_body)
        if response[1]['status'] != 200:
            self.module.fail_json(msg='Could not set the member status! ' + self.host + ' ' + response[1]['status'])
    attributes = property(get_member_attributes)
    status = property(get_member_status, set_member_status)