from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1alpha3CrawlerConfig(_messages.Message):
    """Crawler configuration.

  Fields:
    adHocRun: Ad-hoc option. User can choose this option for ad-hoc runs.
    bucketScope: Bucket scope. Directs the crawler to crawl specific buckets
      within the project that owns the crawler.
    bundleSpecs: The bundling specifications. Directs the crawler to bundle
      files into filesets based on the bundling specifications.
    organizationScope: Organization scope. Directs the crawler to crawl the
      buckets of the projects in the organization that owns the crawler.
    projectScope: Project scope. Directs the crawler to crawl the buckets of
      the project that owns the crawler.
    scheduledRun: Scheduled option. User can choose this option for scheduled
      runs.
  """
    adHocRun = _messages.MessageField('GoogleCloudDatacatalogV1alpha3AdhocRun', 1)
    bucketScope = _messages.MessageField('GoogleCloudDatacatalogV1alpha3BucketScope', 2)
    bundleSpecs = _messages.StringField(3, repeated=True)
    organizationScope = _messages.MessageField('GoogleCloudDatacatalogV1alpha3ParentOrganizationScope', 4)
    projectScope = _messages.MessageField('GoogleCloudDatacatalogV1alpha3ParentProjectScope', 5)
    scheduledRun = _messages.MessageField('GoogleCloudDatacatalogV1alpha3ScheduledRun', 6)