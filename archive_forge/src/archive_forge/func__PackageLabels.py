from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
import six
def _PackageLabels(labels_cls, labels):
    return labels_cls(additionalProperties=[labels_cls.AdditionalProperty(key=key, value=value) for key, value in sorted(six.iteritems(labels))])