from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import contextlib
import dataclasses
import functools
import random
import string
from apitools.base.py import encoding
from apitools.base.py import exceptions as api_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.run import condition as run_condition
from googlecloudsdk.api_lib.run import configuration
from googlecloudsdk.api_lib.run import domain_mapping
from googlecloudsdk.api_lib.run import execution
from googlecloudsdk.api_lib.run import global_methods
from googlecloudsdk.api_lib.run import job
from googlecloudsdk.api_lib.run import metric_names
from googlecloudsdk.api_lib.run import revision
from googlecloudsdk.api_lib.run import route
from googlecloudsdk.api_lib.run import service
from googlecloudsdk.api_lib.run import task
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import apis_internal
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.command_lib.run import artifact_registry
from googlecloudsdk.command_lib.run import config_changes as config_changes_mod
from googlecloudsdk.command_lib.run import exceptions as serverless_exceptions
from googlecloudsdk.command_lib.run import messages_util
from googlecloudsdk.command_lib.run import name_generator
from googlecloudsdk.command_lib.run import op_pollers
from googlecloudsdk.command_lib.run import resource_name_conversion
from googlecloudsdk.command_lib.run import stages
from googlecloudsdk.command_lib.run.sourcedeploys import deployer
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import retry
import six
def _GetBaseRevision(self, template, metadata, status):
    """Return a Revision for use as the "base revision" for a change.

    When making a change that should not affect the code running, the
    "base revision" is the revision that we should lock the code to - it's where
    we get the digest for the image to run.

    Getting this revision:
      * If there's a name in the template metadata, use that
      * If there's a nonce in the revisonTemplate metadata, use that
      * If that query produces >1 or 0 after a short timeout, use
        the latestCreatedRevision in status.

    Arguments:
      template: Revision, the revision template to get the base revision of. May
        have been derived from a Service.
      metadata: ObjectMeta, the metadata from the top-level object
      status: Union[ConfigurationStatus, ServiceStatus], the status of the top-
        level object.

    Returns:
      The base revision of the configuration or None if not found by revision
        name nor nonce and latestCreatedRevisionName does not exist on the
        Service object.
    """
    base_revision = None
    base_revision_name = template.name
    if base_revision_name:
        try:
            revision_ref_getter = functools.partial(self._registry.Parse, params={'namespacesId': metadata.namespace}, collection='run.namespaces.revisions')
            poller = op_pollers.RevisionNameBasedPoller(self, revision_ref_getter)
            base_revision = poller.GetResult(waiter.PollUntilDone(poller, base_revision_name, sleep_ms=500, max_wait_ms=2000))
        except retry.RetryException:
            pass
    if not base_revision:
        base_revision_nonce = template.labels.get(revision.NONCE_LABEL, None)
        if base_revision_nonce:
            try:
                try:
                    namespace_ref = self._registry.Parse(metadata.namespace, collection='run.namespaces')
                except resources.InvalidCollectionException:
                    namespace_ref = self._registry.Parse(metadata.namespace, collection='run.api.v1.namespaces')
                poller = op_pollers.NonceBasedRevisionPoller(self, namespace_ref)
                base_revision = poller.GetResult(waiter.PollUntilDone(poller, base_revision_nonce, sleep_ms=500, max_wait_ms=2000))
            except retry.RetryException:
                pass
    if not base_revision:
        if status.latestCreatedRevisionName:
            revision_ref = self._registry.Parse(status.latestCreatedRevisionName, params={'namespacesId': metadata.namespace}, collection='run.namespaces.revisions')
            base_revision = self.GetRevision(revision_ref)
    return base_revision