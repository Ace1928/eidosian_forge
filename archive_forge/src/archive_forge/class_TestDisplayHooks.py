from holoviews import Curve, Store
from holoviews.ipython import IPTestCase, notebook_extension
class TestDisplayHooks(IPTestCase):

    def setUp(self):
        super().setUp()
        if not notebook_extension._loaded:
            notebook_extension('matplotlib', ip=self.ip)
        self.backup = Store.display_formats
        Store.display_formats = self.format

    def tearDown(self):
        Store._custom_options = {k: {} for k in Store._custom_options.keys()}
        self.ip.run_line_magic('unload_ext', 'holoviews.ipython')
        del self.ip
        Store.display_hooks = self.backup
        notebook_extension._loaded = False
        super().tearDown()