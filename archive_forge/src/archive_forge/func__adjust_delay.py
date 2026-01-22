import logging
from scrapy import signals
from scrapy.exceptions import NotConfigured
def _adjust_delay(self, slot, latency, response):
    """Define delay adjustment policy"""
    target_delay = latency / self.target_concurrency
    new_delay = (slot.delay + target_delay) / 2.0
    new_delay = max(target_delay, new_delay)
    new_delay = min(max(self.mindelay, new_delay), self.maxdelay)
    if response.status != 200 and new_delay <= slot.delay:
        return
    slot.delay = new_delay