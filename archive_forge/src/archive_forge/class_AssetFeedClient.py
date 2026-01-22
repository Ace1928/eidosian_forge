from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import exceptions
from googlecloudsdk.calliope import exceptions as gcloud_exceptions
from googlecloudsdk.command_lib.asset import utils as asset_utils
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.args import repeated
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import times
import six
class AssetFeedClient(object):
    """Client for asset feed."""

    def __init__(self, parent, api_version=DEFAULT_API_VERSION):
        self.parent = parent
        self.message_module = GetMessages(api_version)
        self.service = GetClient(api_version).feeds

    def Create(self, args):
        """Create a feed."""
        content_type = ContentTypeTranslation(args.content_type)
        content_type = getattr(self.message_module.Feed.ContentTypeValueValuesEnum, content_type)
        feed_output_config = self.message_module.FeedOutputConfig(pubsubDestination=self.message_module.PubsubDestination(topic=args.pubsub_topic))
        feed_condition = self.message_module.Expr(expression=args.condition_expression, title=args.condition_title, description=args.condition_description)
        feed = self.message_module.Feed(assetNames=args.asset_names, assetTypes=args.asset_types, contentType=content_type, feedOutputConfig=feed_output_config, condition=feed_condition, relationshipTypes=args.relationship_types)
        create_feed_request = self.message_module.CreateFeedRequest(feed=feed, feedId=args.feed)
        request_message = self.message_module.CloudassetFeedsCreateRequest(parent=self.parent, createFeedRequest=create_feed_request)
        return self.service.Create(request_message)

    def Describe(self, args):
        """Describe a feed."""
        request_message = self.message_module.CloudassetFeedsGetRequest(name='{}/feeds/{}'.format(self.parent, args.feed))
        return self.service.Get(request_message)

    def Delete(self, args):
        """Delete a feed."""
        request_message = self.message_module.CloudassetFeedsDeleteRequest(name='{}/feeds/{}'.format(self.parent, args.feed))
        self.service.Delete(request_message)

    def List(self):
        """List feeds under a parent."""
        request_message = self.message_module.CloudassetFeedsListRequest(parent=self.parent)
        return self.service.List(request_message)

    def Update(self, args):
        """Update a feed."""
        update_masks = []
        content_type = ContentTypeTranslation(args.content_type)
        content_type = getattr(self.message_module.Feed.ContentTypeValueValuesEnum, content_type)
        feed_name = '{}/feeds/{}'.format(self.parent, args.feed)
        if args.content_type or args.clear_content_type:
            update_masks.append('content_type')
        if args.pubsub_topic:
            update_masks.append('feed_output_config.pubsub_destination.topic')
        if args.condition_expression or args.clear_condition_expression:
            update_masks.append('condition.expression')
        if args.condition_title or args.clear_condition_title:
            update_masks.append('condition.title')
        if args.condition_description or args.clear_condition_description:
            update_masks.append('condition.description')
        asset_names, asset_types, relationship_types = self.UpdateAssetNamesTypesAndRelationships(args, feed_name, update_masks)
        update_mask = ','.join(update_masks)
        feed_output_config = self.message_module.FeedOutputConfig(pubsubDestination=self.message_module.PubsubDestination(topic=args.pubsub_topic))
        feed_condition = self.message_module.Expr(expression=args.condition_expression, title=args.condition_title, description=args.condition_description)
        feed = self.message_module.Feed(assetNames=asset_names, assetTypes=asset_types, contentType=content_type, feedOutputConfig=feed_output_config, condition=feed_condition, relationshipTypes=relationship_types)
        update_feed_request = self.message_module.UpdateFeedRequest(feed=feed, updateMask=update_mask)
        request_message = self.message_module.CloudassetFeedsPatchRequest(name=feed_name, updateFeedRequest=update_feed_request)
        return self.service.Patch(request_message)

    def UpdateAssetNamesTypesAndRelationships(self, args, feed_name, update_masks):
        """Get Updated assetNames, assetTypes and relationshipTypes."""
        feed = self.service.Get(self.message_module.CloudassetFeedsGetRequest(name=feed_name))
        asset_names = repeated.ParsePrimitiveArgs(args, 'asset_names', lambda: feed.assetNames)
        if asset_names is not None:
            update_masks.append('asset_names')
        else:
            asset_names = []
        asset_types = repeated.ParsePrimitiveArgs(args, 'asset_types', lambda: feed.assetTypes)
        if asset_types is not None:
            update_masks.append('asset_types')
        else:
            asset_types = []
        relationship_types = repeated.ParsePrimitiveArgs(args, 'relationship_types', lambda: feed.relationshipTypes)
        if relationship_types is not None:
            update_masks.append('relationship_types')
        else:
            relationship_types = []
        return (asset_names, asset_types, relationship_types)