from typing import Any, cast, Dict, TYPE_CHECKING, Union
from gitlab import cli
from gitlab import exceptions as exc
from gitlab import types
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import CRUDMixin, ObjectDeleteMixin, SaveMixin
from gitlab.types import RequiredOptional
Merge two topics, assigning all projects to the target topic.

        Args:
            source_topic_id: ID of source project topic
            target_topic_id: ID of target project topic
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabTopicMergeError: If the merge failed

        Returns:
            The merged topic data (*not* a RESTObject)
        