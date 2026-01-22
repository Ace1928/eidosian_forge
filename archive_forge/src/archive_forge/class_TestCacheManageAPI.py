from unittest import mock
from glance.api.v2 import cached_images
from glance import notifier
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
class TestCacheManageAPI(test_utils.BaseTestCase):

    def setUp(self):
        super(TestCacheManageAPI, self).setUp()
        self.req = unit_test_utils.get_fake_request()

    def _main_test_helper(self, argv, status='active', image_mock=True):
        with mock.patch.object(notifier.ImageRepoProxy, 'get') as mock_get:
            image = FakeImage(status=status)
            mock_get.return_value = image
            with mock.patch.object(cached_images.CacheController, '_enforce') as e:
                with mock.patch('glance.image_cache.ImageCache') as ic:
                    cc = cached_images.CacheController()
                    cc.cache = ic
                    c_calls = []
                    c_calls += argv[0].split(',')
                    for call in c_calls:
                        mock.patch.object(ic, call)
                    test_call = getattr(cc, argv[1])
                    new_policy = argv[2]
                    args = []
                    if len(argv) == 4:
                        args = argv[3:]
                    test_call(self.req, *args)
                    if image_mock:
                        e.assert_called_once_with(self.req, image=image, new_policy=new_policy)
                    else:
                        e.assert_called_once_with(self.req, new_policy=new_policy)
                    mcs = []
                    for method in ic.method_calls:
                        mcs.append(str(method))
                    for call in c_calls:
                        if args == []:
                            args.append('')
                        elif args[0] and (not args[0].endswith("'")):
                            args[0] = "'" + args[0] + "'"
                        self.assertIn('call.' + call + '(' + args[0] + ')', mcs)
                    self.assertEqual(len(c_calls), len(mcs))

    def test_delete_cache_entry(self):
        self._main_test_helper(['delete_cached_image,delete_queued_image', 'delete_cache_entry', 'cache_delete', UUID1])

    def test_clear_cache(self):
        self._main_test_helper(['delete_all_cached_images,delete_all_queued_images', 'clear_cache', 'cache_delete'], image_mock=False)

    def test_get_cache_state(self):
        self._main_test_helper(['get_cached_images,get_queued_images', 'get_cache_state', 'cache_list'], image_mock=False)

    @mock.patch.object(cached_images, 'WORKER')
    def test_queue_image_from_api(self, mock_worker):
        self._main_test_helper(['queue_image', 'queue_image_from_api', 'cache_image', UUID1])
        mock_worker.submit.assert_called_once_with(UUID1)

    def test_init_no_config(self):
        self.assertIsNone(cached_images.WORKER)
        self.config(image_cache_dir=None)
        cached_images.CacheController()
        self.assertIsNone(cached_images.WORKER)

    def test_init_with_config(self):
        self.assertIsNone(cached_images.WORKER)
        self.config(image_cache_dir='/tmp')
        cached_images.CacheController()
        self.assertIsNotNone(cached_images.WORKER)
        self.assertTrue(cached_images.WORKER.is_alive())
        cached_images.WORKER.terminate()