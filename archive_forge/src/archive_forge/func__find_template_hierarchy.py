import os
import json
from jupyter_core.paths import jupyter_path
import nbconvert.exporters.templateexporter
def _find_template_hierarchy(app_names, template_name, root_dirs):
    template_names = []
    while template_name is not None:
        template_names.append(template_name)
        conf = {}
        for root_dir in root_dirs:
            for app_name in app_names:
                conf_file = os.path.join(root_dir, app_name, 'templates', template_name, 'conf.json')
                if os.path.exists(conf_file):
                    with open(conf_file) as f:
                        new_conf = json.load(f)
                        new_conf.update(conf)
                        conf = new_conf
        if 'base_template' in conf:
            template_name = conf['base_template']
        elif template_name == 'base':
            template_name = None
        else:
            template_name = 'base'
    return template_names