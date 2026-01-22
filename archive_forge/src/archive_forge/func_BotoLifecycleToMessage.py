from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import datetime
import json
import re
import textwrap
import xml.etree.ElementTree
from xml.etree.ElementTree import ParseError as XmlParseError
import six
from apitools.base.protorpclite.util import decode_datetime
from apitools.base.py import encoding
import boto
from boto.gs.acl import ACL
from boto.gs.acl import ALL_AUTHENTICATED_USERS
from boto.gs.acl import ALL_USERS
from boto.gs.acl import Entries
from boto.gs.acl import Entry
from boto.gs.acl import GROUP_BY_DOMAIN
from boto.gs.acl import GROUP_BY_EMAIL
from boto.gs.acl import GROUP_BY_ID
from boto.gs.acl import USER_BY_EMAIL
from boto.gs.acl import USER_BY_ID
from boto.s3.tagging import Tags
from boto.s3.tagging import TagSet
from gslib.cloud_api import ArgumentException
from gslib.cloud_api import BucketNotFoundException
from gslib.cloud_api import NotFoundException
from gslib.cloud_api import Preconditions
from gslib.exception import CommandException
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.utils.constants import S3_ACL_MARKER_GUID
from gslib.utils.constants import S3_MARKER_GUIDS
@classmethod
def BotoLifecycleToMessage(cls, boto_lifecycle):
    """Translates a boto lifecycle object to an apitools message."""
    lifecycle_message = None
    if boto_lifecycle:
        lifecycle_message = apitools_messages.Bucket.LifecycleValue()
        for boto_rule in boto_lifecycle:
            lifecycle_rule = apitools_messages.Bucket.LifecycleValue.RuleValueListEntry()
            lifecycle_rule.condition = apitools_messages.Bucket.LifecycleValue.RuleValueListEntry.ConditionValue()
            if boto_rule.action:
                if boto_rule.action == boto.gs.lifecycle.DELETE:
                    lifecycle_rule.action = apitools_messages.Bucket.LifecycleValue.RuleValueListEntry.ActionValue(type='Delete')
                elif boto_rule.action == boto.gs.lifecycle.SET_STORAGE_CLASS:
                    lifecycle_rule.action = apitools_messages.Bucket.LifecycleValue.RuleValueListEntry.ActionValue(type='SetStorageClass', storageClass=boto_rule.action_text)
            if boto.gs.lifecycle.AGE in boto_rule.conditions:
                lifecycle_rule.condition.age = int(boto_rule.conditions[boto.gs.lifecycle.AGE])
            if boto.gs.lifecycle.CREATED_BEFORE in boto_rule.conditions:
                lifecycle_rule.condition.createdBefore = LifecycleTranslation.TranslateBotoLifecycleTimestamp(boto_rule.conditions[boto.gs.lifecycle.CREATED_BEFORE])
            if boto.gs.lifecycle.IS_LIVE in boto_rule.conditions:
                boto_is_live_str = boto_rule.conditions[boto.gs.lifecycle.IS_LIVE].lower()
                if boto_is_live_str == 'true':
                    lifecycle_rule.condition.isLive = True
                elif boto_is_live_str == 'false':
                    lifecycle_rule.condition.isLive = False
                else:
                    raise CommandException('Got an invalid Boto value for IsLive condition ("%s"), expected "true" or "false".' % boto_rule.conditions[boto.gs.lifecycle.IS_LIVE])
            if boto.gs.lifecycle.MATCHES_STORAGE_CLASS in boto_rule.conditions:
                for storage_class in boto_rule.conditions[boto.gs.lifecycle.MATCHES_STORAGE_CLASS]:
                    lifecycle_rule.condition.matchesStorageClass.append(storage_class)
            if boto.gs.lifecycle.NUM_NEWER_VERSIONS in boto_rule.conditions:
                lifecycle_rule.condition.numNewerVersions = int(boto_rule.conditions[boto.gs.lifecycle.NUM_NEWER_VERSIONS])
            lifecycle_message.rule.append(lifecycle_rule)
    return lifecycle_message