import os
from traitlets import Unicode
from ipywidgets import Widget
from .VueComponentRegistry import vue_component_files, register_component_from_file
from ._version import semver
class VueEventHandler(FileSystemEventHandler):

    def on_modified(self, event):
        super(VueEventHandler, self).on_modified(event)
        if not event.is_directory:
            if event.src_path in template_registry:
                log.info(f'updating: {event.src_path}')
                with open(event.src_path) as f:
                    template_registry[event.src_path].template = f.read()
            elif event.src_path in vue_component_files:
                log.info(f'updating component: {event.src_path}')
                name = vue_component_files[event.src_path]
                register_component_from_file(name, event.src_path)