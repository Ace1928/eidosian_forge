from tensorboard.plugins import base_plugin
class _ProfileRedirectPlugin(base_plugin.TBPlugin):
    """Redirect notice pointing users to the new dynamic profile plugin."""
    plugin_name = 'profile_redirect'

    def get_plugin_apps(self):
        return {}

    def is_active(self):
        return False

    def frontend_metadata(self):
        return base_plugin.FrontendMetadata(element_name='tf-profile-redirect-dashboard', tab_name='Profile')