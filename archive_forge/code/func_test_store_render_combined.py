from holoviews import Curve, Store
from holoviews.ipython import IPTestCase, notebook_extension
def test_store_render_combined(self):
    curve = Curve([1, 2, 3])
    data, metadata = Store.render(curve)
    mime_types = {'text/html', 'image/svg+xml', 'image/png'}
    self.assertEqual(set(data), mime_types)