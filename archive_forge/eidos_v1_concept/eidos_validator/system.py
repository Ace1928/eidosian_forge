"""
System Information Collection
--------------------------
Collects system-related information including hardware, OS, and resources.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

from .cache import CacheManager
from .network import NetworkInfoCollector
from .utils import cache_key

# Get logger
logger = logging.getLogger("eidos_validator.system")


class SystemInfoCollector:
    """
    Collects comprehensive system information with fallbacks.

    This class gathers information about the host system including hardware,
    operating system, available resources, and constraints. It uses a cache
    to minimize the overhead of collecting this information repeatedly.
    """

    def __init__(
        self,
        cache_manager: Optional[CacheManager] = None,
        network_collector: Optional[NetworkInfoCollector] = None,
    ) -> None:
        """
        Initialize the system info collector.

        Args:
            cache_manager: Optional CacheManager instance to use for caching
            network_collector: Optional NetworkInfoCollector for network info
        """
        self.cache_manager = cache_manager or CacheManager()
        self.network_collector = network_collector or NetworkInfoCollector(
            self.cache_manager
        )
        self.available_modules: Dict[str, bool] = {}

    def set_available_modules(self, modules: Dict[str, bool]) -> None:
        """
        Set the dictionary of available modules.

        Args:
            modules: Dictionary mapping module names to availability status
        """
        self.available_modules = modules
        self.network_collector.set_available_modules(modules)

    def get_system_info(self) -> Dict[str, Any]:
        """
        Gather comprehensive system information with fallbacks and detailed logging.

        Returns:
            Dict[str, Any]: Dictionary of system information organized by category
        """
        logger.info("Gathering system information...")
        cache_key_static = "system_static"
        cache_key_dynamic = "system_dynamic"
        cache_key_volatile = "system_volatile"

        # Try to get cached static info
        static_info = self.cache_manager.get(cache_key_static)
        if static_info:
            info = static_info.copy()
            logger.debug("Using cached static system info")
        else:
            logger.debug("Building new static system info")
            info = self._build_static_system_info()

            # Cache static info
            self.cache_manager.set(cache_key_static, info, "static")
            logger.debug("Cached static system info")

        # Get dynamic info (updated hourly) with logging
        dynamic_info = self.cache_manager.get(cache_key_dynamic)
        if not dynamic_info:
            logger.debug("Building new dynamic system info")
            dynamic_info = self._build_dynamic_system_info()

            # Cache dynamic info
            self.cache_manager.set(cache_key_dynamic, dynamic_info, "dynamic")
            logger.debug("Cached dynamic system info")

        # Get volatile info (updated every 5 min) with logging
        volatile_info = self.cache_manager.get(cache_key_volatile)
        if not volatile_info:
            logger.debug("Building new volatile system info")
            volatile_info = self._build_volatile_system_info()

            # Cache volatile info
            self.cache_manager.set(cache_key_volatile, volatile_info, "volatile")
            logger.debug("Cached volatile system info")

        # Merge all info
        info.update(dynamic_info)
        info.update(volatile_info)

        logger.info("System information gathering complete")
        return info

    def _build_static_system_info(self) -> Dict[str, Any]:
        """
        Build static system information that rarely changes.

        Returns:
            Dict[str, Any]: Dictionary of static system information
        """
        info: Dict[str, Any] = {
            "computational_resources": {
                "processing_power": "Unknown",
                "memory_capacity": "Unknown",
                "storage_capacity": "Unknown",
                "network_bandwidth": "Unknown",
                "specialized_hardware": [],
                "cpu_details": {},
                "memory_details": {},
                "storage_details": {},
            },
            "operating_system": {
                "name": "Unknown",
                "version": "Unknown",
                "architecture": "Unknown",
                "kernel": "Unknown",
            },
            "hardware": {
                "machine": "Unknown",
                "processor": "Unknown",
                "gpus": [],
                "other_devices": [],
            },
            "network_connectivity": "Unknown",
            "external_systems": [],
            "constraints": [],
            "available_apis": [],
            "security_protocols": [],
            "data_sources": [],
            "environment_variables": {},
            "python_info": {
                "version": sys.version,
                "implementation": sys.implementation.name,
                "available_modules": [
                    k for k, v in self.available_modules.items() if v
                ],
            },
        }

        # Get static hardware info with logging
        self._collect_cpu_info(info)
        self._collect_memory_info(info)
        self._collect_storage_info(info)
        self._collect_gpu_info(info)
        self._collect_os_info(info)
        self._collect_environment_variables(info)

        return info

    def _build_dynamic_system_info(self) -> Dict[str, Any]:
        """
        Build dynamic system information that changes periodically.

        Returns:
            Dict[str, Any]: Dictionary of dynamic system information
        """
        dynamic_info: Dict[str, Any] = {}

        # Get network info
        network_info = self.network_collector.get_network_info()
        dynamic_info.update(network_info)
        logger.debug("Added network info to dynamic system info")

        # Get available APIs
        ml_libs = ["numpy", "tensorflow", "torch", "pandas", "sklearn"]
        web_libs = ["requests", "aiohttp", "urllib3"]
        db_libs = ["sqlite3", "pymongo", "sqlalchemy"]

        apis = []
        for lib_list in [ml_libs, web_libs, db_libs]:
            for lib in lib_list:
                if lib in self.available_modules and self.available_modules[lib]:
                    apis.append(lib)
        dynamic_info["available_apis"] = apis
        logger.debug(f"Added {len(apis)} available APIs")

        return dynamic_info

    def _build_volatile_system_info(self) -> Dict[str, Any]:
        """
        Build volatile system information that changes frequently.

        Returns:
            Dict[str, Any]: Dictionary of volatile system information
        """
        volatile_info: Dict[str, Any] = {}

        if self.available_modules.get("psutil", False):
            try:
                import psutil

                # CPU usage with logging
                cpu_info = {
                    "frequency": (
                        psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {}
                    ),
                    "usage_percent": psutil.cpu_percent(interval=1),
                    "stats": psutil.cpu_stats()._asdict(),
                }

                # Ensure 'computational_resources' and 'storage_details' keys exist
                if "computational_resources" not in volatile_info:
                    volatile_info["computational_resources"] = {}
                if "storage_details" not in volatile_info["computational_resources"]:
                    volatile_info["computational_resources"]["storage_details"] = {}

                volatile_info["computational_resources"]["cpu_details"] = cpu_info
                logger.debug(f"Added CPU usage: {cpu_info['usage_percent']}%")

                # Memory usage with logging
                mem = psutil.virtual_memory()
                swap = psutil.swap_memory()
                volatile_info["computational_resources"]["memory_details"] = {
                    "available": mem.available,
                    "used": mem.used,
                    "free": mem.free,
                    "swap": swap._asdict(),
                }
                logger.debug(
                    f"Added memory usage - Available: {mem.available / (1024**3):.1f}GB"
                )

                # Storage usage with logging
                partitions = []
                for part in psutil.disk_partitions():
                    try:
                        usage = psutil.disk_usage(part.mountpoint)
                        partitions.append({"used": usage.used, "free": usage.free})
                        logger.debug(f"Added storage usage for {part.mountpoint}")
                    except Exception as e:
                        logger.warning(
                            f"Failed to get usage for {part.mountpoint}: {str(e)}"
                        )
                        continue
                volatile_info["computational_resources"]["storage_details"][
                    "usage"
                ] = partitions

                # System constraints with logging
                constraints = []
                if mem.available < 4 * 1024**3:  # 4GB
                    constraints.append("Low available memory")
                    logger.warning("Low memory condition detected")
                if psutil.cpu_percent(interval=1) > 80:
                    constraints.append("High CPU usage")
                    logger.warning("High CPU usage detected")
                if any(
                    psutil.disk_usage(p.mountpoint).percent > 90
                    for p in psutil.disk_partitions()
                ):
                    constraints.append("Low disk space")
                    logger.warning("Low disk space condition detected")
                volatile_info["constraints"] = constraints
                if constraints:
                    logger.warning(f"System constraints detected: {constraints}")

            except Exception as e:
                logger.error(f"Error getting volatile info: {str(e)}")

        return volatile_info

    def _collect_cpu_info(self, info: Dict[str, Any]) -> None:
        """
        Collect CPU information.

        Args:
            info: Dictionary to update with CPU info
        """
        if self.available_modules.get("psutil", False):
            try:
                import psutil

                cpu_info = {
                    "physical_cores": psutil.cpu_count(logical=False),
                    "logical_cores": psutil.cpu_count(logical=True),
                }
                info["computational_resources"]["cpu_details"].update(cpu_info)
                if self.available_modules.get("platform", False):
                    import platform

                    info["computational_resources"][
                        "processing_power"
                    ] = f"{cpu_info['logical_cores']} cores ({platform.processor()})"
                else:
                    info["computational_resources"][
                        "processing_power"
                    ] = f"{cpu_info['logical_cores']} cores (Unknown processor)"
                logger.debug(
                    f"Added CPU info: {cpu_info['logical_cores']} logical cores"
                )
            except Exception as e:
                logger.error(f"Failed to get CPU info: {str(e)}")

    def _collect_memory_info(self, info: Dict[str, Any]) -> None:
        """
        Collect memory information.

        Args:
            info: Dictionary to update with memory info
        """
        if self.available_modules.get("psutil", False):
            try:
                import psutil

                mem = psutil.virtual_memory()
                info["computational_resources"][
                    "memory_capacity"
                ] = f"{round(mem.total / (1024**3))}GB RAM"
                logger.debug(
                    f"Added memory capacity: {info['computational_resources']['memory_capacity']}"
                )
            except Exception as e:
                logger.error(f"Failed to get memory info: {str(e)}")

    def _collect_storage_info(self, info: Dict[str, Any]) -> None:
        """
        Collect storage information.

        Args:
            info: Dictionary to update with storage info
        """
        if self.available_modules.get("psutil", False):
            try:
                import psutil

                partitions = []
                total = 0
                for part in psutil.disk_partitions():
                    try:
                        usage = psutil.disk_usage(part.mountpoint)
                        partitions.append(
                            {
                                "device": part.device,
                                "mountpoint": part.mountpoint,
                                "fstype": part.fstype,
                                "total": usage.total,
                            }
                        )
                        total += usage.total
                        logger.debug(
                            f"Added partition: {part.device} ({usage.total / (1024**3):.1f}GB)"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to get usage for partition {part.device}: {str(e)}"
                        )
                        continue
                info["computational_resources"]["storage_details"][
                    "partitions"
                ] = partitions
                info["computational_resources"][
                    "storage_capacity"
                ] = f"{round(total / (1024**3))}GB Total"
                logger.debug(
                    f"Total storage capacity: {info['computational_resources']['storage_capacity']}"
                )
            except Exception as e:
                logger.error(f"Failed to get storage info: {str(e)}")

    def _collect_gpu_info(self, info: Dict[str, Any]) -> None:
        """
        Collect GPU information.

        Args:
            info: Dictionary to update with GPU info
        """
        if self.available_modules.get("subprocess", False):
            try:
                import subprocess

                nvidia_smi = subprocess.check_output(["nvidia-smi", "-L"]).decode(
                    "utf-8"
                )
                for line in nvidia_smi.strip().split("\n"):
                    info["hardware"]["gpus"].append(
                        {"type": "NVIDIA", "info": line.strip()}
                    )
                logger.debug(f"Found NVIDIA GPUs: {len(info['hardware']['gpus'])}")
            except Exception as e:
                logger.debug(f"No NVIDIA GPUs found: {str(e)}")
                if self.available_modules.get(
                    "platform", False
                ) and self.available_modules.get("subprocess", False):
                    import platform

                    if platform.system() == "Linux":
                        try:
                            import subprocess

                            lspci = (
                                subprocess.check_output(["lspci"])
                                .decode("utf-8")
                                .lower()
                            )
                            for line in lspci.split("\n"):
                                if "amd" in line and (
                                    "radeon" in line or "gpu" in line
                                ):
                                    info["hardware"]["gpus"].append(
                                        {"type": "AMD", "info": line.strip()}
                                    )
                            if info["hardware"]["gpus"]:
                                logger.debug(
                                    f"Found AMD GPUs: {len(info['hardware']['gpus'])}"
                                )
                        except Exception as e:
                            logger.debug(f"No AMD GPUs found: {str(e)}")

    def _collect_os_info(self, info: Dict[str, Any]) -> None:
        """
        Collect operating system information.

        Args:
            info: Dictionary to update with OS info
        """
        if self.available_modules.get("platform", False):
            try:
                import platform

                os_info = f"{platform.system()} {platform.release()}"
                info["operating_system"].update(
                    {
                        "name": platform.system(),
                        "version": platform.release(),
                        "architecture": platform.machine(),
                        "kernel": platform.version(),
                    }
                )
                logger.debug(f"Added OS info: {os_info}")
            except Exception as e:
                logger.error(f"Failed to get OS info: {str(e)}")

    def _collect_environment_variables(self, info: Dict[str, Any]) -> None:
        """
        Collect environment variables.

        Args:
            info: Dictionary to update with environment variables
        """
        if self.available_modules.get("os", False):
            try:
                import os

                info["environment_variables"] = dict(os.environ)
                logger.debug(
                    f"Added {len(info['environment_variables'])} environment variables"
                )
            except Exception as e:
                logger.error(f"Failed to get environment variables: {str(e)}")
