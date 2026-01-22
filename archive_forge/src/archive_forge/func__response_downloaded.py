import logging
from scrapy import signals
from scrapy.exceptions import NotConfigured
def _response_downloaded(self, response, request, spider):
    key, slot = self._get_slot(request, spider)
    latency = request.meta.get('download_latency')
    if latency is None or slot is None:
        return
    olddelay = slot.delay
    self._adjust_delay(slot, latency, response)
    if self.debug:
        diff = slot.delay - olddelay
        size = len(response.body)
        conc = len(slot.transferring)
        logger.info('slot: %(slot)s | conc:%(concurrency)2d | delay:%(delay)5d ms (%(delaydiff)+d) | latency:%(latency)5d ms | size:%(size)6d bytes', {'slot': key, 'concurrency': conc, 'delay': slot.delay * 1000, 'delaydiff': diff * 1000, 'latency': latency * 1000, 'size': size}, extra={'spider': spider})