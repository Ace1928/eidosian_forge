import json
import logging
from http.client import RemoteDisconnected
from ray.autoscaler.node_provider import NodeProvider
from ray.autoscaler.tags import TAG_RAY_CLUSTER_NAME
NodeProvider for automatically managed private/local clusters.

    The cluster management is handled by a remote coordinating server.
    The server listens on <coordinator_address>, therefore, the address
    should be provided in the provider section in the cluster config.
    The server receieves HTTP requests from this class and uses
    LocalNodeProvider to get their responses.
    