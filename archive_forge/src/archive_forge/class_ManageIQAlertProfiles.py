from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.manageiq import ManageIQ, manageiq_argument_spec
class ManageIQAlertProfiles(object):
    """ Object to execute alert profile management operations in manageiq.
    """

    def __init__(self, manageiq):
        self.manageiq = manageiq
        self.module = self.manageiq.module
        self.api_url = self.manageiq.api_url
        self.client = self.manageiq.client
        self.url = '{api_url}/alert_definition_profiles'.format(api_url=self.api_url)

    def get_profiles(self):
        """ Get all alert profiles from ManageIQ
        """
        try:
            response = self.client.get(self.url + '?expand=alert_definitions,resources')
        except Exception as e:
            self.module.fail_json(msg='Failed to query alert profiles: {error}'.format(error=e))
        return response.get('resources') or []

    def get_alerts(self, alert_descriptions):
        """ Get a list of alert hrefs from a list of alert descriptions
        """
        alerts = []
        for alert_description in alert_descriptions:
            alert = self.manageiq.find_collection_resource_or_fail('alert_definitions', description=alert_description)
            alerts.append(alert['href'])
        return alerts

    def add_profile(self, profile):
        """ Add a new alert profile to ManageIQ
        """
        alerts = self.get_alerts(profile['alerts'])
        profile_dict = dict(name=profile['name'], description=profile['name'], mode=profile['resource_type'])
        if profile['notes']:
            profile_dict['set_data'] = dict(notes=profile['notes'])
        try:
            result = self.client.post(self.url, resource=profile_dict, action='create')
        except Exception as e:
            self.module.fail_json(msg='Creating profile failed {error}'.format(error=e))
        self.assign_or_unassign(result['results'][0], alerts, 'assign')
        msg = 'Profile {name} created successfully'
        msg = msg.format(name=profile['name'])
        return dict(changed=True, msg=msg)

    def delete_profile(self, profile):
        """ Delete an alert profile from ManageIQ
        """
        try:
            self.client.post(profile['href'], action='delete')
        except Exception as e:
            self.module.fail_json(msg='Deleting profile failed: {error}'.format(error=e))
        msg = 'Successfully deleted profile {name}'.format(name=profile['name'])
        return dict(changed=True, msg=msg)

    def get_alert_href(self, alert):
        """ Get an absolute href for an alert
        """
        return '{url}/alert_definitions/{id}'.format(url=self.api_url, id=alert['id'])

    def assign_or_unassign(self, profile, resources, action):
        """ Assign or unassign alerts to profile, and validate the result.
        """
        alerts = [dict(href=href) for href in resources]
        subcollection_url = profile['href'] + '/alert_definitions'
        try:
            result = self.client.post(subcollection_url, resources=alerts, action=action)
            if len(result['results']) != len(alerts):
                msg = "Failed to {action} alerts to profile '{name}'," + 'expected {expected} alerts to be {action}ed,' + 'but only {changed} were {action}ed'
                msg = msg.format(action=action, name=profile['name'], expected=len(alerts), changed=result['results'])
                self.module.fail_json(msg=msg)
        except Exception as e:
            msg = "Failed to {action} alerts to profile '{name}': {error}"
            msg = msg.format(action=action, name=profile['name'], error=e)
            self.module.fail_json(msg=msg)
        return result['results']

    def update_profile(self, old_profile, desired_profile):
        """ Update alert profile in ManageIQ
        """
        changed = False
        old_profile = self.client.get(old_profile['href'] + '?expand=alert_definitions')
        desired_alerts = set(self.get_alerts(desired_profile['alerts']))
        if 'alert_definitions' in old_profile:
            existing_alerts = set([self.get_alert_href(alert) for alert in old_profile['alert_definitions']])
        else:
            existing_alerts = set()
        to_add = list(desired_alerts - existing_alerts)
        to_remove = list(existing_alerts - desired_alerts)
        if to_remove:
            self.assign_or_unassign(old_profile, to_remove, 'unassign')
            changed = True
        if to_add:
            self.assign_or_unassign(old_profile, to_add, 'assign')
            changed = True
        profile_dict = dict()
        if old_profile['mode'] != desired_profile['resource_type']:
            profile_dict['mode'] = desired_profile['resource_type']
        old_notes = old_profile.get('set_data', {}).get('notes')
        if desired_profile['notes'] != old_notes:
            profile_dict['set_data'] = dict(notes=desired_profile['notes'])
        if profile_dict:
            changed = True
            try:
                result = self.client.post(old_profile['href'], resource=profile_dict, action='edit')
            except Exception as e:
                msg = "Updating profile '{name}' failed: {error}"
                msg = msg.format(name=old_profile['name'], error=e)
                self.module.fail_json(msg=msg)
        if changed:
            msg = 'Profile {name} updated successfully'.format(name=desired_profile['name'])
        else:
            msg = 'No update needed for profile {name}'.format(name=desired_profile['name'])
        return dict(changed=changed, msg=msg)