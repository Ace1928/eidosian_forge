from kivy.app import App
class UnicodeIssueSetting(App):

    def build_config(self, config):
        config.add_section('test')
        config.setdefault('test', 'string', 'Hello world')
        config.setdefault('test', 'path', '/')

    def build(self):
        from kivy.uix.settings import Settings
        s = Settings()
        s.add_json_panel('Test Panel', self.config, data=data)
        return s