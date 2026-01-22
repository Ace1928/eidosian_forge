from django.contrib.syndication.views import Feed as BaseFeed
from django.utils.feedgenerator import Atom1Feed, Rss201rev2Feed
def add_item_elements(self, handler, item):
    super().add_item_elements(handler, item)
    self.add_georss_element(handler, item, w3c_geo=True)