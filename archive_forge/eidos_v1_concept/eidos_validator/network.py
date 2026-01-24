"""
Network Information Collection
---------------------------
Collects network-related information for the system.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

from .cache import CacheManager
from .utils import cache_key

# Get logger
logger = logging.getLogger("eidos_validator.network")


class NetworkInfoCollector:
    """
    Collects network information with comprehensive fallbacks.
    """

    def __init__(self, cache_manager: Optional[CacheManager] = None) -> None:
        """
        Initialize the network info collector.

        Args:
            cache_manager: Optional CacheManager instance to use for caching
        """
        self.cache_manager = cache_manager or CacheManager()
        self.available_modules = {}

    def set_available_modules(self, modules: Dict[str, bool]) -> None:
        """
        Set the dictionary of available modules.

        Args:
            modules: Dictionary mapping module names to availability status
        """
        self.available_modules = modules

    def get_network_info(self) -> Dict[str, Any]:
        """
        Gather network information with comprehensive fallbacks and logging.

        Returns:
            Dict[str, Any]: Dictionary of network information
        """
        logger.info("Gathering network information...")
        cache_key_static = "network_static"
        cache_key_dynamic = "network_dynamic"

        # Try to get cached static info
        static_info = self.cache_manager.get(cache_key_static)
        if static_info:
            info = static_info.copy()
            logger.debug("Using cached static network info")
        else:
            logger.debug("Building new static network info")
            info = {
                "network_connectivity": "Unknown",
                "external_systems": [],
                "available_apis": [],
                "security_protocols": [],
                "network_interfaces": [],
                "network_stats": {},
                "data_sources": [],  # Required by schema
            }

            # Get static network info with detailed logging
            self._collect_hostname_info(info)
            self._collect_security_protocols(info)
            self._collect_network_interfaces(info)

            # Cache static info
            self.cache_manager.set(cache_key_static, info, "static")
            logger.debug("Cached static network info")

        # Get dynamic network info
        dynamic_info = self.cache_manager.get(cache_key_dynamic)
        if not dynamic_info:
            logger.debug("Building new dynamic network info")
            dynamic_info = {}

            self._test_connectivity(dynamic_info)
            self._collect_network_stats(dynamic_info)

            # Cache dynamic info
            self.cache_manager.set(cache_key_dynamic, dynamic_info, "dynamic")
            logger.debug("Cached dynamic network info")

        # Merge dynamic into static info
        info.update(dynamic_info)
        logger.info("Network information gathering complete")
        return info

    def _collect_hostname_info(self, info: Dict[str, Any]) -> None:
        """
        Collect hostname and IP information.

        Args:
            info: Dictionary to update with collected info
        """
        if self.available_modules.get("socket", False):
            try:
                import socket

                hostname = socket.gethostname()
                ip = socket.gethostbyname(hostname)
                info["external_systems"].extend([f"Hostname: {hostname}", f"IP: {ip}"])
                logger.debug(f"Added hostname and IP info: {hostname}, {ip}")
            except Exception as e:
                logger.error(f"Failed to get hostname/IP: {str(e)}")

    def _collect_security_protocols(self, info: Dict[str, Any]) -> None:
        """
        Collect security protocol information.

        Args:
            info: Dictionary to update with collected info
        """
        if self.available_modules.get("ssl", False):
            try:
                import ssl

                protocols = []
                if hasattr(ssl, "PROTOCOL_TLSv1_2"):
                    protocols.append("TLSv1.2")
                if hasattr(ssl, "PROTOCOL_TLSv1_3"):
                    protocols.append("TLSv1.3")
                info["security_protocols"].extend(protocols)
                logger.debug(f"Added SSL protocols: {protocols}")
            except Exception as e:
                logger.error(f"Failed to get SSL protocols: {str(e)}")

    def _collect_network_interfaces(self, info: Dict[str, Any]) -> None:
        """
        Collect network interface information.

        Args:
            info: Dictionary to update with collected info
        """
        if self.available_modules.get("psutil", False):
            try:
                import psutil

                net_if = psutil.net_if_addrs()
                for interface, addrs in net_if.items():
                    if_info = {"name": interface, "addresses": []}
                    for addr in addrs:
                        if_info["addresses"].append(str(addr.address))
                    info["network_interfaces"].append(if_info)
                logger.debug(f"Added network interfaces: {len(net_if)} found")
            except Exception as e:
                logger.error(f"Failed to get network interfaces: {str(e)}")

    def _test_connectivity(self, info: Dict[str, Any]) -> None:
        """
        Test internet connectivity.

        Args:
            info: Dictionary to update with collected info
        """
        if self.available_modules.get("requests", False):
            import requests

            test_urls = [
                "http://google.com",
                "http://cloudflare.com",
                "http://amazon.com",
            ]
            for url in test_urls:
                try:
                    logger.debug(f"Testing connectivity to {url}")
                    requests.get(url, timeout=3)
                    info["network_connectivity"] = "Connected"
                    logger.debug(f"Successfully connected to {url}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to connect to {url}: {str(e)}")
                    continue

            if "network_connectivity" not in info:
                info["network_connectivity"] = "Disconnected"
                logger.warning("All connectivity tests failed")

    def _collect_network_stats(self, info: Dict[str, Any]) -> None:
        """
        Collect network statistics.

        Args:
            info: Dictionary to update with collected info
        """
        if self.available_modules.get("psutil", False):
            try:
                import psutil

                net_io = psutil.net_io_counters()
                info["network_stats"] = {
                    "bytes_sent": net_io.bytes_sent,
                    "bytes_recv": net_io.bytes_recv,
                    "packets_sent": net_io.packets_sent,
                    "packets_recv": net_io.packets_recv,
                }
                logger.debug("Added network I/O stats")
            except Exception as e:
                logger.error(f"Failed to get network stats: {str(e)}")
