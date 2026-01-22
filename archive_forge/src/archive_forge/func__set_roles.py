from paste.util import ip4
def _set_roles(self, environ, roles):
    cur_roles = environ.get('REMOTE_USER_TOKENS', '').split(',')
    cur_roles = list(filter(None, cur_roles))
    remove_roles = []
    for role in roles:
        if role.startswith('-'):
            remove_roles.append(role[1:])
        elif role not in cur_roles:
            cur_roles.append(role)
    for role in remove_roles:
        if role in cur_roles:
            cur_roles.remove(role)
    environ['REMOTE_USER_TOKENS'] = ','.join(cur_roles)