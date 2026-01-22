from tensorboard.plugins import base_plugin
class _WITRedirectPlugin(base_plugin.TBPlugin):
    """Redirect notice pointing users to the new dynamic LIT plugin."""
    plugin_name = 'wit_redirect'

    def get_plugin_apps(self):
        return {}

    def is_active(self):
        return False

    def frontend_metadata(self):
        return base_plugin.FrontendMetadata(element_name='tf-wit-redirect-dashboard', tab_name='What-If Tool')