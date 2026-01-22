import glance_store
from oslo_concurrency import lockutils
from oslo_config import cfg
from oslo_log import log as logging
from glance.api import common as api_common
from glance.common import exception
from glance import context
from glance.i18n import _LI, _LW
from glance.image_cache import base
class Prefetcher(base.CacheApp):

    def __init__(self):
        import glance.gateway
        super(Prefetcher, self).__init__()
        self.gateway = glance.gateway.Gateway()

    def fetch_image_into_cache(self, image_id):
        ctx = context.RequestContext(is_admin=True, show_deleted=True, roles=['admin'])
        try:
            image_repo = self.gateway.get_repo(ctx)
            image = image_repo.get(image_id)
        except exception.NotFound:
            LOG.warning(_LW("Image '%s' not found"), image_id)
            return False
        if image.status != 'active':
            LOG.warning(_LW("Image '%s' is not active. Not caching."), image_id)
            return False
        for loc in image.locations:
            if CONF.enabled_backends:
                image_data, image_size = glance_store.get(loc['url'], None, context=ctx)
            else:
                image_data, image_size = glance_store.get_from_backend(loc['url'], context=ctx)
            LOG.debug("Caching image '%s'", image_id)
            cache_tee_iter = self.cache.cache_tee_iter(image_id, image_data, image.checksum)
            list(cache_tee_iter)
            return True

    @lockutils.lock('glance-cache', external=True)
    def run(self):
        images = self.cache.get_queued_images()
        if not images:
            LOG.debug('Nothing to prefetch.')
            return True
        num_images = len(images)
        LOG.debug('Found %d images to prefetch', num_images)
        pool = api_common.get_thread_pool('prefetcher', size=num_images)
        results = pool.map(self.fetch_image_into_cache, images)
        successes = sum([1 for r in results if r is True])
        if successes != num_images:
            LOG.warning(_LW('Failed to successfully cache all images in queue.'))
            return False
        LOG.info(_LI('Successfully cached all %d images'), num_images)
        return True