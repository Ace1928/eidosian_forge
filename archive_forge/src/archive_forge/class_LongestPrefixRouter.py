import logging
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple
from ray.serve._private.common import (
from ray.serve._private.constants import (
from ray.serve.handle import RayServeHandle
class LongestPrefixRouter(ProxyRouter):
    """Router that performs longest prefix matches on incoming routes."""

    def __init__(self, get_handle: Callable, protocol: RequestProtocol):
        self._get_handle = get_handle
        self._protocol = protocol
        self.sorted_routes: List[str] = list()
        self.route_info: Dict[str, EndpointTag] = dict()
        self.handles: Dict[EndpointTag, RayServeHandle] = dict()
        self.app_to_is_cross_language: Dict[ApplicationName, bool] = dict()

    def update_routes(self, endpoints: Dict[EndpointTag, EndpointInfo]) -> None:
        logger.info(f'Got updated endpoints: {endpoints}.', extra={'log_to_stderr': False})
        existing_handles = set(self.handles.keys())
        routes = []
        route_info = {}
        app_to_is_cross_language = {}
        for endpoint, info in endpoints.items():
            routes.append(info.route)
            route_info[info.route] = endpoint
            app_to_is_cross_language[endpoint.app] = info.app_is_cross_language
            if endpoint in self.handles:
                existing_handles.remove(endpoint)
            else:
                handle = self._get_handle(endpoint.name, endpoint.app).options(stream=not info.app_is_cross_language, use_new_handle_api=True, _prefer_local_routing=RAY_SERVE_PROXY_PREFER_LOCAL_NODE_ROUTING)
                handle._set_request_protocol(self._protocol)
                self.handles[endpoint] = handle
        if len(existing_handles) > 0:
            logger.info(f'Deleting {len(existing_handles)} unused handles.', extra={'log_to_stderr': False})
        for endpoint in existing_handles:
            del self.handles[endpoint]
        self.sorted_routes = sorted(routes, key=lambda x: len(x), reverse=True)
        self.route_info = route_info
        self.app_to_is_cross_language = app_to_is_cross_language

    def match_route(self, target_route: str) -> Optional[Tuple[str, RayServeHandle, bool]]:
        """Return the longest prefix match among existing routes for the route.
        Args:
            target_route: route to match against.
        Returns:
            (route, handle, is_cross_language) if found, else None.
        """
        for route in self.sorted_routes:
            if target_route.startswith(route):
                matched = False
                if route.endswith('/'):
                    matched = True
                elif len(target_route) == len(route) or target_route[len(route)] == '/':
                    matched = True
                if matched:
                    endpoint = self.route_info[route]
                    return (route, self.handles[endpoint], self.app_to_is_cross_language[endpoint.app])
        return None