from __future__ import (absolute_import, division, print_function)
def yaml_to_dict(yaml, content_id):
    """
            Return a Python dict version of the provided YAML.
            Conversion is done in a subprocess since the current Python interpreter does not have access to PyYAML.
            """
    if content_id in yaml_to_dict_cache:
        return yaml_to_dict_cache[content_id]
    try:
        cmd = [external_python, yaml_to_json_path]
        proc = subprocess.Popen([to_bytes(c) for c in cmd], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout_bytes, stderr_bytes = proc.communicate(to_bytes(yaml))
        if proc.returncode != 0:
            raise Exception('command %s failed with return code %d: %s' % ([to_native(c) for c in cmd], proc.returncode, to_native(stderr_bytes)))
        data = yaml_to_dict_cache[content_id] = json.loads(to_text(stdout_bytes), object_hook=object_hook)
        return data
    except Exception as ex:
        raise Exception('internal importer error - failed to parse yaml: %s' % to_native(ex))