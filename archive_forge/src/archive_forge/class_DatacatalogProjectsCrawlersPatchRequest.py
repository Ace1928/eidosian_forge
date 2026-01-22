from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatacatalogProjectsCrawlersPatchRequest(_messages.Message):
    """A DatacatalogProjectsCrawlersPatchRequest object.

  Fields:
    googleCloudDatacatalogV1alpha3Crawler: A
      GoogleCloudDatacatalogV1alpha3Crawler resource to be passed as the
      request body.
    name: Output only. The resource name of the crawler. Must be empty when
      creating a crawler. For example, "projects/a/crawlers/b".
    updateMask: The update mask applies to the resource.
  """
    googleCloudDatacatalogV1alpha3Crawler = _messages.MessageField('GoogleCloudDatacatalogV1alpha3Crawler', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)