from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.heroku import HerokuHelper
def add_or_delete_heroku_collaborator(module, client):
    user = module.params['user']
    state = module.params['state']
    affected_apps = []
    result_state = False
    for app in module.params['apps']:
        if app not in client.apps():
            module.fail_json(msg='App {0} does not exist'.format(app))
        heroku_app = client.apps()[app]
        heroku_collaborator_list = [collaborator.user.email for collaborator in heroku_app.collaborators()]
        if state == 'absent' and user in heroku_collaborator_list:
            if not module.check_mode:
                heroku_app.remove_collaborator(user)
            affected_apps += [app]
            result_state = True
        elif state == 'present' and user not in heroku_collaborator_list:
            if not module.check_mode:
                heroku_app.add_collaborator(user_id_or_email=user, silent=module.params['suppress_invitation'])
            affected_apps += [app]
            result_state = True
    return (result_state, affected_apps)