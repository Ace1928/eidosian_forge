from __future__ import absolute_import, division, print_function
from os import path
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils import deps
def add_user_to_teams(self, pd_user_id, pd_teams, pd_role):
    updated_team = None
    for team in pd_teams:
        team_info = self._apisession.find('teams', team, attribute='name')
        if team_info is not None:
            try:
                updated_team = self._apisession.rput('/teams/' + team_info['id'] + '/users/' + pd_user_id, json={'role': pd_role})
            except PDClientError:
                updated_team = None
    return updated_team