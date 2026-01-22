import re
def _do_to_path(structure, path, command):
    if not path:
        return command(structure) if callable(command) else command
    kvs = _get_keys_and_values(structure, path[0])
    return _update_structure(structure, kvs, path[1:], command)