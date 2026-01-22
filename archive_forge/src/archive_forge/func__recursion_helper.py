from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import abc
import enum
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import folder_util
from googlecloudsdk.command_lib.storage import plurality_checkable_iterator
from googlecloudsdk.command_lib.storage import wildcard_iterator
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.command_lib.storage.resources import shim_format_util
import six
def _recursion_helper(self, iterator, recursion_level, print_top_level_container=True):
    """For retrieving resources from URLs that potentially contain wildcards.

    Args:
      iterator (Iterable[resource_reference.Resource]): For recursing through.
      recursion_level (int): Integer controlling how deep the listing recursion
        goes. "1" is the default, mimicking the actual OS ls, which lists the
        contents of the first level of matching subdirectories. Call with
        "float('inf')" for listing everything available.
      print_top_level_container (bool): Used by `du` to skip printing the top
        level bucket

    Yields:
      BaseFormatWrapper generator.
    """
    for resource in iterator:
        if resource_reference.is_container_or_has_container_url(resource) and recursion_level > 0:
            if self._header_wrapper != NullFormatWrapper:
                yield self._header_wrapper(resource, display_detail=self._display_detail, include_etag=self._include_etag, object_state=self._object_state, readable_sizes=self._readable_sizes, full_formatter=self._full_formatter, use_gsutil_style=self._use_gsutil_style)
            container_size = 0
            nested_iterator = self._get_container_iterator(resource.storage_url, recursion_level - 1)
            for nested_resource in nested_iterator:
                if self._container_summary_wrapper != NullFormatWrapper and print_top_level_container:
                    container_size += getattr(nested_resource.resource, 'size', 0)
                yield nested_resource
            if self._container_summary_wrapper != NullFormatWrapper and print_top_level_container:
                yield self._container_summary_wrapper(resource=resource, container_size=container_size, object_state=self._object_state, readable_sizes=self._readable_sizes)
        else:
            yield self._object_wrapper(resource, display_detail=self._display_detail, full_formatter=self._full_formatter, include_etag=self._include_etag, object_state=self._object_state, readable_sizes=self._readable_sizes, use_gsutil_style=self._use_gsutil_style)