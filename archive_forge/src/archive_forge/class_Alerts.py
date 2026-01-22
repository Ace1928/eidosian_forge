from __future__ import absolute_import, division, print_function
import json
import logging
from pprint import pformat
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import request, eseries_host_argument_spec
from ansible.module_utils._text import to_native
class Alerts(object):

    def __init__(self):
        argument_spec = eseries_host_argument_spec()
        argument_spec.update(dict(state=dict(type='str', required=False, default='enabled', choices=['enabled', 'disabled']), server=dict(type='str', required=False), sender=dict(type='str', required=False), contact=dict(type='str', required=False), recipients=dict(type='list', required=False), test=dict(type='bool', required=False, default=False), log_path=dict(type='str', required=False)))
        required_if = [['state', 'enabled', ['server', 'sender', 'recipients']]]
        self.module = AnsibleModule(argument_spec=argument_spec, supports_check_mode=True, required_if=required_if)
        args = self.module.params
        self.alerts = args['state'] == 'enabled'
        self.server = args['server']
        self.sender = args['sender']
        self.contact = args['contact']
        self.recipients = args['recipients']
        self.test = args['test']
        self.ssid = args['ssid']
        self.url = args['api_url']
        self.creds = dict(url_password=args['api_password'], validate_certs=args['validate_certs'], url_username=args['api_username'])
        self.check_mode = self.module.check_mode
        log_path = args['log_path']
        self._logger = logging.getLogger(self.__class__.__name__)
        if log_path:
            logging.basicConfig(level=logging.DEBUG, filename=log_path, filemode='w', format='%(relativeCreated)dms %(levelname)s %(module)s.%(funcName)s:%(lineno)d\n %(message)s')
        if not self.url.endswith('/'):
            self.url += '/'
        email = re.compile('[^@]+@[^@]+\\.[^@]+')
        if self.sender and (not email.match(self.sender)):
            self.module.fail_json(msg='The sender (%s) provided is not a valid email address.' % self.sender)
        if self.recipients is not None:
            for recipient in self.recipients:
                if not email.match(recipient):
                    self.module.fail_json(msg='The recipient (%s) provided is not a valid email address.' % recipient)
            if len(self.recipients) < 1:
                self.module.fail_json(msg='At least one recipient address must be specified.')

    def get_configuration(self):
        try:
            rc, result = request(self.url + 'storage-systems/%s/device-alerts' % self.ssid, headers=HEADERS, **self.creds)
            self._logger.info('Current config: %s', pformat(result))
            return result
        except Exception as err:
            self.module.fail_json(msg='Failed to retrieve the alerts configuration! Array Id [%s]. Error [%s].' % (self.ssid, to_native(err)))

    def update_configuration(self):
        config = self.get_configuration()
        update = False
        body = dict()
        if self.alerts:
            body = dict(alertingEnabled=True)
            if not config['alertingEnabled']:
                update = True
            body.update(emailServerAddress=self.server)
            if config['emailServerAddress'] != self.server:
                update = True
            body.update(additionalContactInformation=self.contact, sendAdditionalContactInformation=True)
            if self.contact and (self.contact != config['additionalContactInformation'] or not config['sendAdditionalContactInformation']):
                update = True
            body.update(emailSenderAddress=self.sender)
            if config['emailSenderAddress'] != self.sender:
                update = True
            self.recipients.sort()
            if config['recipientEmailAddresses']:
                config['recipientEmailAddresses'].sort()
            body.update(recipientEmailAddresses=self.recipients)
            if config['recipientEmailAddresses'] != self.recipients:
                update = True
        elif config['alertingEnabled']:
            body = dict(alertingEnabled=False)
            update = True
        self._logger.debug(pformat(body))
        if update and (not self.check_mode):
            try:
                rc, result = request(self.url + 'storage-systems/%s/device-alerts' % self.ssid, method='POST', data=json.dumps(body), headers=HEADERS, **self.creds)
            except Exception as err:
                self.module.fail_json(msg='We failed to set the storage-system name! Array Id [%s]. Error [%s].' % (self.ssid, to_native(err)))
        return update

    def send_test_email(self):
        """Send a test email to verify that the provided configuration is valid and functional."""
        if not self.check_mode:
            try:
                rc, result = request(self.url + 'storage-systems/%s/device-alerts/alert-email-test' % self.ssid, timeout=300, method='POST', headers=HEADERS, **self.creds)
                if result['response'] != 'emailSentOK':
                    self.module.fail_json(msg='The test email failed with status=[%s]! Array Id [%s].' % (result['response'], self.ssid))
            except Exception as err:
                self.module.fail_json(msg='We failed to send the test email! Array Id [%s]. Error [%s].' % (self.ssid, to_native(err)))

    def update(self):
        update = self.update_configuration()
        if self.test and update:
            self._logger.info('An update was detected and test=True, running a test.')
            self.send_test_email()
        if self.alerts:
            msg = 'Alerting has been enabled using server=%s, sender=%s.' % (self.server, self.sender)
        else:
            msg = 'Alerting has been disabled.'
        self.module.exit_json(msg=msg, changed=update)

    def __call__(self, *args, **kwargs):
        self.update()