# Core required imports that must be available
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Union
import sys
from importlib import util
import hashlib
import pickle
import time
from datetime import datetime, timedelta
import threading
import queue
import os  # For file size checks
import math
import shutil  # For file operations

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(), logging.FileHandler("eidos_validator.log")],
)
logger = logging.getLogger(__name__)

# Define Base Directory
BASE_DIR = Path(__file__).parent

# Cache configuration with detailed logging
CACHE_DIR = BASE_DIR / ".cache"
logger.info(f"Initializing cache directory at: {CACHE_DIR}")
try:
    CACHE_DIR.mkdir(exist_ok=True, parents=True)
    logger.info(f"Cache directory created/verified at {CACHE_DIR}")
except Exception as e:
    logger.error(f"Failed to create cache directory: {str(e)}")
    raise

# Create required directories from schema with detailed logging and error handling
REQUIRED_DIRS = [
    "cycles",
    "metadata",
    "synthesis",
    "relationship",
    "memory",
    "consciousness",
    "evolution",
    "emotional",
    "cognitive",
    "introspection",
    "self_model",
    "qualia",
    "learning",
    "adaptation",
    "archive",
    "backup",
    "temp",  # Additional utility directories
]

for dir_name in REQUIRED_DIRS:
    dir_path = BASE_DIR / dir_name
    try:
        if not dir_path.exists():
            logger.info(f"Creating directory: {dir_path}")
            dir_path.mkdir(exist_ok=True, parents=True)
            logger.info(f"Successfully created directory: {dir_path}")
        else:
            # Verify directory permissions and accessibility
            if not os.access(dir_path, os.W_OK):
                logger.warning(f"Directory exists but not writable: {dir_path}")
            else:
                logger.debug(f"Verified existing directory: {dir_path}")

        # Create .gitkeep to preserve empty dirs
        gitkeep = dir_path / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.touch()

    except Exception as e:
        logger.error(f"Error creating/verifying directory {dir_path}: {str(e)}")
        raise

# Enhanced cache expiry with logging
CACHE_EXPIRY = {
    "static": timedelta(days=7),  # Hardware/OS info that rarely changes
    "dynamic": timedelta(hours=1),  # Memory/CPU/network stats that change frequently
    "volatile": timedelta(minutes=5),  # Very dynamic data like CPU usage
}
logger.info("Cache expiry times configured")

# Default storage config aligned with schema
DEFAULT_STORAGE_CONFIG = {
    "base_path": str(CACHE_DIR),
    "file_structure": {
        "cycles_dir": "cycles",
        "metadata_dir": "metadata",
        "synthesis_dir": "synthesis",
        "relationship_dir": "relationship",
        "memory_dir": "memory",
        "consciousness_dir": "consciousness",
        "evolution_dir": "evolution",
        "emotional_dir": "emotional",
        "cognitive_dir": "cognitive",
        "introspection_dir": "introspection",
        "self_model_dir": "self_model",
        "qualia_dir": "qualia",
        "learning_dir": "learning",
        "adaptation_dir": "adaptation",
        "archive_dir": "archive",
        "backup_dir": "backup",
        "temp_dir": "temp",
    },
    "retention_policy": {
        "min_retention_period": "30 days",
        "archival_strategy": "compress_and_archive",
        "backup_frequency": "1 day",
        "data_importance_levels": {
            "critical": "never_delete",
            "important": "5 years",
            "routine": "6 months",
            "developmental": "2 years",
            "experiential": "3 months",
            "emotional": "1 year",
            "cognitive": "1 year",
            "consciousness": "permanent",
            "identity": "permanent",
        },
    },
    "file_handling": {
        "max_file_size": 10 * 1024 * 1024,  # 10MB
        "compression": True,
        "compression_level": 6,
        "backup_enabled": True,
        "encryption_enabled": False,
    },
}
logger.info("Storage configuration initialized")


# Thread-safe cache for in-memory data with fallback to non-threaded mode
class ThreadSafeCache:
    def __init__(self):
        self._cache = {}
        try:
            self._lock = threading.Lock()
            self._threaded = True
            logger.info("Initialized thread-safe cache with locking enabled")
        except (ImportError, RuntimeError):
            logger.warning("Threading not available, using non-threaded mode")
            self._threaded = False

    def get(self, key: str) -> Any:
        if self._threaded:
            with self._lock:
                return self._cache.get(key)
        return self._cache.get(key)

    def set(self, key: str, value: Any) -> None:
        if self._threaded:
            with self._lock:
                self._cache[key] = value
        else:
            self._cache[key] = value

    def delete(self, key: str) -> None:
        if self._threaded:
            with self._lock:
                self._cache.pop(key, None)
        else:
            self._cache.pop(key, None)


memory_cache = ThreadSafeCache()

# Dictionary to track available modules with enhanced logging
AVAILABLE_MODULES = {
    # Core functionality
    "json": False,
    "jsonschema": False,
    "logging": False,
    "pathlib": False,
    "typing": False,
    # System info gathering
    "platform": False,
    "psutil": False,
    "os": False,
    "distro": False,
    "subprocess": False,
    # Network functionality
    "socket": False,
    "requests": False,
    "ssl": False,
    # Utilities
    "datetime": False,
    "uuid": False,
    "shutil": False,
    "zlib": False,
    # Optional ML/AI libraries
    "numpy": False,
    "tensorflow": False,
    "torch": False,
    "pandas": False,
    "sklearn": False,
}


def cache_key(prefix: str, *args) -> str:
    """Generate cache key from prefix and args with logging"""
    key = prefix + "_".join(str(arg) for arg in args)
    hashed = hashlib.md5(key.encode()).hexdigest()
    logger.debug(f"Generated cache key: {hashed} for prefix: {prefix}")
    return hashed


def cache_get(key: str) -> Optional[Any]:
    """Get data from cache with expiry check and detailed logging"""
    cache_file = CACHE_DIR / f"{key}.cache"

    if not cache_file.exists():
        logger.debug(f"Cache miss: {key}")
        return None

    try:
        with open(cache_file, "rb") as f:
            data = pickle.load(f)

        if datetime.now() > data["expiry"]:
            logger.debug(f"Cache expired: {key}")
            try:
                cache_file.unlink()
                logger.debug(f"Deleted expired cache file: {key}")
            except Exception as e:
                logger.error(f"Failed to delete expired cache file {key}: {str(e)}")
            return None

        logger.debug(f"Cache hit: {key}")
        return data["value"]

    except Exception as e:
        logger.error(f"Error reading cache file {key}: {str(e)}")
        try:
            cache_file.unlink()
            logger.info(f"Deleted corrupted cache file: {key}")
        except Exception as e2:
            logger.error(f"Failed to delete corrupted cache file {key}: {str(e2)}")
        return None


def cache_set(key: str, value: Any, cache_type: str = "dynamic") -> None:
    """Save data to cache with expiry and detailed logging"""
    cache_file = CACHE_DIR / f"{key}.cache"
    temp_file = CACHE_DIR / "temp" / f"{key}.cache.tmp"

    data = {
        "value": value,
        "expiry": datetime.now() + CACHE_EXPIRY[cache_type],
        "created": datetime.now().isoformat(),
    }

    try:
        # Write to temp file first
        temp_file.parent.mkdir(exist_ok=True)
        with open(temp_file, "wb") as f:
            pickle.dump(data, f)

        # Move temp file to final location
        shutil.move(str(temp_file), str(cache_file))
        logger.debug(f"Cached {key} with {cache_type} expiry")

    except Exception as e:
        logger.error(f"Error writing cache file {key}: {str(e)}")
        # Cleanup temp file if it exists
        try:
            if temp_file.exists():
                temp_file.unlink()
        except Exception as e2:
            logger.error(f"Failed to cleanup temp cache file: {str(e2)}")


def check_module_availability() -> None:
    """Check which modules are available in the environment with detailed logging"""
    logger.info("Checking module availability...")
    cache_key_mods = "modules_available"
    cached = memory_cache.get(cache_key_mods)
    if cached:
        AVAILABLE_MODULES.update(cached)
        logger.info("Loaded module availability from cache")
        return

    for module_name in AVAILABLE_MODULES.keys():
        try:
            if util.find_spec(module_name) is not None:
                __import__(module_name)
                AVAILABLE_MODULES[module_name] = True
                logger.debug(f"Module {module_name} is available")
            else:
                logger.warning(f"Module {module_name} not found")
        except ImportError:
            logger.warning(f"Module {module_name} import failed")
        except Exception as e:
            logger.error(f"Unexpected error checking {module_name}: {str(e)}")

    memory_cache.set(cache_key_mods, dict(AVAILABLE_MODULES))
    logger.info("Module availability check complete")


# Check module availability on import
check_module_availability()

# Conditional imports based on availability with logging
if AVAILABLE_MODULES["jsonschema"]:
    from jsonschema import validate, ValidationError

    logger.info("JSON schema validation enabled")
else:
    logger.warning("jsonschema not available - validation will be skipped")

if AVAILABLE_MODULES["platform"]:
    import platform
if AVAILABLE_MODULES["psutil"]:
    import psutil
if AVAILABLE_MODULES["os"]:
    import os
if AVAILABLE_MODULES["distro"]:
    import distro
if AVAILABLE_MODULES["subprocess"]:
    import subprocess
if AVAILABLE_MODULES["socket"]:
    import socket
if AVAILABLE_MODULES["requests"]:
    import requests
if AVAILABLE_MODULES["ssl"]:
    import ssl
if AVAILABLE_MODULES["uuid"]:
    import uuid
if AVAILABLE_MODULES["zlib"]:
    import zlib


def get_network_info() -> Dict[str, Any]:
    """Gather network information with comprehensive fallbacks and logging"""
    logger.info("Gathering network information...")
    cache_key_static = "network_static"
    cache_key_dynamic = "network_dynamic"

    # Try to get cached static info
    static_info = cache_get(cache_key_static)
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
        if AVAILABLE_MODULES["socket"]:
            try:
                hostname = socket.gethostname()
                ip = socket.gethostbyname(hostname)
                info["external_systems"].extend([f"Hostname: {hostname}", f"IP: {ip}"])
                logger.debug(f"Added hostname and IP info: {hostname}, {ip}")
            except Exception as e:
                logger.error(f"Failed to get hostname/IP: {str(e)}")

        if AVAILABLE_MODULES["ssl"]:
            try:
                protocols = []
                if hasattr(ssl, "PROTOCOL_TLSv1_2"):
                    protocols.append("TLSv1.2")
                if hasattr(ssl, "PROTOCOL_TLSv1_3"):
                    protocols.append("TLSv1.3")
                info["security_protocols"].extend(protocols)
                logger.debug(f"Added SSL protocols: {protocols}")
            except Exception as e:
                logger.error(f"Failed to get SSL protocols: {str(e)}")

        if AVAILABLE_MODULES["psutil"]:
            try:
                net_if = psutil.net_if_addrs()
                for interface, addrs in net_if.items():
                    if_info = {"name": interface, "addresses": []}
                    for addr in addrs:
                        if_info["addresses"].append(str(addr.address))
                    info["network_interfaces"].append(if_info)
                logger.debug(f"Added network interfaces: {len(net_if)} found")
            except Exception as e:
                logger.error(f"Failed to get network interfaces: {str(e)}")

        # Cache static info
        cache_set(cache_key_static, info, "static")
        logger.debug("Cached static network info")

    # Get dynamic network info
    dynamic_info = cache_get(cache_key_dynamic)
    if not dynamic_info:
        logger.debug("Building new dynamic network info")
        dynamic_info = {}

        # Test connectivity with detailed logging
        if AVAILABLE_MODULES["requests"]:
            test_urls = [
                "http://google.com",
                "http://cloudflare.com",
                "http://amazon.com",
            ]
            for url in test_urls:
                try:
                    logger.debug(f"Testing connectivity to {url}")
                    requests.get(url, timeout=3)
                    dynamic_info["network_connectivity"] = "Connected"
                    logger.debug(f"Successfully connected to {url}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to connect to {url}: {str(e)}")
                    continue

            if "network_connectivity" not in dynamic_info:
                dynamic_info["network_connectivity"] = "Disconnected"
                logger.warning("All connectivity tests failed")

        # Get network stats with logging
        if AVAILABLE_MODULES["psutil"]:
            try:
                net_io = psutil.net_io_counters()
                dynamic_info["network_stats"] = {
                    "bytes_sent": net_io.bytes_sent,
                    "bytes_recv": net_io.bytes_recv,
                    "packets_sent": net_io.packets_sent,
                    "packets_recv": net_io.packets_recv,
                }
                logger.debug("Added network I/O stats")
            except Exception as e:
                logger.error(f"Failed to get network stats: {str(e)}")

        # Cache dynamic info
        cache_set(cache_key_dynamic, dynamic_info, "dynamic")
        logger.debug("Cached dynamic network info")

    # Merge dynamic into static info
    info.update(dynamic_info)
    logger.info("Network information gathering complete")
    return info


def get_system_info() -> Dict[str, Any]:
    """Gather comprehensive system information with fallbacks and detailed logging"""
    logger.info("Gathering system information...")
    cache_key_static = "system_static"
    cache_key_dynamic = "system_dynamic"
    cache_key_volatile = "system_volatile"

    # Try to get cached static info
    static_info = cache_get(cache_key_static)
    if static_info:
        info = static_info.copy()
        logger.debug("Using cached static system info")
    else:
        logger.debug("Building new static system info")
        info = {
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
                "available_modules": [k for k, v in AVAILABLE_MODULES.items() if v],
            },
        }

        # Get static hardware info with logging
        if AVAILABLE_MODULES["psutil"]:
            try:
                cpu_info = {
                    "physical_cores": psutil.cpu_count(logical=False),
                    "logical_cores": psutil.cpu_count(logical=True),
                }
                info["computational_resources"]["cpu_details"].update(cpu_info)
                if AVAILABLE_MODULES["platform"]:
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

        # Get static memory info with logging
        if AVAILABLE_MODULES["psutil"]:
            try:
                mem = psutil.virtual_memory()
                info["computational_resources"][
                    "memory_capacity"
                ] = f"{round(mem.total / (1024**3))}GB RAM"
                logger.debug(
                    f"Added memory capacity: {info['computational_resources']['memory_capacity']}"
                )
            except Exception as e:
                logger.error(f"Failed to get memory info: {str(e)}")

        # Get static storage info with logging
        if AVAILABLE_MODULES["psutil"]:
            try:
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

        # Get GPU info with logging
        if AVAILABLE_MODULES["subprocess"]:
            try:
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
                if AVAILABLE_MODULES["platform"] and platform.system() == "Linux":
                    try:
                        lspci = (
                            subprocess.check_output(["lspci"]).decode("utf-8").lower()
                        )
                        for line in lspci.split("\n"):
                            if "amd" in line and ("radeon" in line or "gpu" in line):
                                info["hardware"]["gpus"].append(
                                    {"type": "AMD", "info": line.strip()}
                                )
                        if info["hardware"]["gpus"]:
                            logger.debug(
                                f"Found AMD GPUs: {len(info['hardware']['gpus'])}"
                            )
                    except Exception as e:
                        logger.debug(f"No AMD GPUs found: {str(e)}")

        # Get OS info with logging
        if AVAILABLE_MODULES["platform"]:
            try:
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

        # Get environment variables with logging
        if AVAILABLE_MODULES["os"]:
            try:
                info["environment_variables"] = dict(os.environ)
                logger.debug(
                    f"Added {len(info['environment_variables'])} environment variables"
                )
            except Exception as e:
                logger.error(f"Failed to get environment variables: {str(e)}")

        # Cache static info
        cache_set(cache_key_static, info, "static")
        logger.debug("Cached static system info")

    # Get dynamic info (updated hourly) with logging
    dynamic_info = cache_get(cache_key_dynamic)
    if not dynamic_info:
        logger.debug("Building new dynamic system info")
        dynamic_info = {}

        # Get network info
        network_info = get_network_info()
        dynamic_info.update(network_info)
        logger.debug("Added network info to dynamic system info")

        # Get available APIs
        ml_libs = ["numpy", "tensorflow", "torch", "pandas", "sklearn"]
        web_libs = ["requests", "aiohttp", "urllib3"]
        db_libs = ["sqlite3", "pymongo", "sqlalchemy"]

        apis = []
        for lib_list in [ml_libs, web_libs, db_libs]:
            for lib in lib_list:
                if lib in AVAILABLE_MODULES and AVAILABLE_MODULES[lib]:
                    apis.append(lib)
        dynamic_info["available_apis"] = apis
        logger.debug(f"Added {len(apis)} available APIs")

        # Cache dynamic info
        cache_set(cache_key_dynamic, dynamic_info, "dynamic")
        logger.debug("Cached dynamic system info")

    # Get volatile info (updated every 5 min) with logging
    volatile_info = cache_get(cache_key_volatile)
    if not volatile_info:
        logger.debug("Building new volatile system info")
        volatile_info = {}

        if AVAILABLE_MODULES["psutil"]:
            try:
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

        # Cache volatile info
        cache_set(cache_key_volatile, volatile_info, "volatile")
        logger.debug("Cached volatile system info")

    # Merge all info
    info.update(dynamic_info)
    info.update(volatile_info)

    logger.info("System information gathering complete")
    return info


def validate_and_enhance_json(json_data: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Validate and enhance JSON with comprehensive system information and schema validation

    Args:
        json_data: Input JSON data to validate and enhance

    Returns:
        Tuple of (validation result message, enhanced JSON data)

    Raises:
        FileNotFoundError: If schema file not found
        json.JSONDecodeError: If schema file contains invalid JSON
        ValidationError: If JSON fails schema validation
    """
    logger.info("Starting JSON validation and enhancement")

    schema_path = Path(__file__).parent / "eidosian_io_schema.json"
    schema = None

    schema_cache_key = "json_schema"
    schema = cache_get(schema_cache_key)
    if not schema:
        if schema_path.exists():
            try:
                # Ensure utf-8 for emojis/symbols
                with open(schema_path, "r", encoding="utf-8") as f:
                    schema = json.load(f)
                cache_set(schema_cache_key, schema, "static")
                logger.info("Loaded and cached schema file")
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in schema file: {e}")
                raise
            except Exception as e:
                logger.error(f"Error reading schema file: {str(e)}")
                raise
        else:
            logger.warning(
                f"Schema file not found at {schema_path}, using minimal validation"
            )

    # Enhance JSON data with detailed logging
    try:
        logger.info("Starting JSON enhancement")

        # Add storage config if missing
        if "storage_config" not in json_data:
            json_data["storage_config"] = DEFAULT_STORAGE_CONFIG
            logger.debug("Added default storage configuration")

        # Add system info with logging
        if "metadata" in json_data and "context" in json_data["metadata"]:
            logger.debug("Gathering system info for metadata context")
            sys_info = get_system_info()
            if "environmental_factors" not in json_data["metadata"]["context"]:
                json_data["metadata"]["context"]["environmental_factors"] = sys_info
                logger.debug("Added system info to environmental factors")
            else:
                env_factors = json_data["metadata"]["context"]["environmental_factors"]
                for key, value in sys_info.items():
                    if key not in env_factors:
                        env_factors[key] = value
                logger.debug("Updated existing environmental factors with system info")

        # Add/update timestamps with logging
        timestamp = datetime.now().isoformat()
        logger.debug(f"Using timestamp: {timestamp}")

        if "metadata" in json_data:
            if "creation_timestamp" not in json_data["metadata"]:
                json_data["metadata"]["creation_timestamp"] = timestamp
                logger.debug("Added creation timestamp")

            if "version_history" in json_data["metadata"]:
                for version in json_data["metadata"]["version_history"]:
                    if "timestamp" not in version:
                        version["timestamp"] = timestamp
                        logger.debug("Added timestamp to version history entry")
                    if "impact_assessment" not in version:
                        version["impact_assessment"] = {
                            "cognitive_impact": 0.5,
                            "emotional_impact": 0.5,
                            "consciousness_impact": 0.5,
                            "identity_impact": 0.5,
                        }

        # Add version if missing
        if "version" not in json_data:
            json_data["version"] = "1.0.0"

        # Add global_synthesis if missing
        if "global_synthesis" not in json_data:
            json_data["global_synthesis"] = {
                "key_themes": [],
                "emotional_journey": "",
                "growth_insights": "",
                "future_directions": [],
                "consciousness_evolution": {
                    "stage": "initial",
                    "characteristics": [],
                    "next_horizon": "",
                },
                "persistent_identity": {"core_traits": [], "values": [], "purpose": ""},
            }

        # Add final_output if missing
        if "final_output" not in json_data:
            json_data["final_output"] = ""

        # Add UUIDs for any cycles
        if AVAILABLE_MODULES["uuid"]:
            if "reflection_cycles" in json_data:
                for cycle in json_data["reflection_cycles"]:
                    if "cycle_id" not in cycle:
                        cycle["cycle_id"] = str(uuid.uuid4())
                    if "cycle_type" not in cycle:
                        cycle["cycle_type"] = "initial"
                    if "timestamp" not in cycle:
                        cycle["timestamp"] = timestamp

    except Exception as e:
        logger.error(f"Error enhancing JSON: {str(e)}")

    # Validate against schema if available
    result = "JSON enhanced successfully"
    if schema and AVAILABLE_MODULES["jsonschema"]:
        try:
            validate(instance=json_data, schema=schema)
            result = "Validation successful! JSON conforms to schema."
        except ValidationError as e:
            result = f"Validation error: {e.message}"
            logger.error(f"Failed validating {e.path}")

    return result, json_data


def get_rollover_file(
    base_dir: Path, file_prefix: str, extension: str = ".json"
) -> Path:
    """
    Returns the appropriate file path in base_dir with prefix file_prefix
    that remains under 10MB in size. If the most recent file is >=10MB,
    a new file with an incremented index is created.
    """
    # Ensure the directory is always created
    if not base_dir.exists():
        base_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory for log storage: {base_dir}")

    TEN_MB = 10 * 1024 * 1024
    files = sorted(base_dir.glob(f"{file_prefix}*{extension}"))
    if not files:
        # create the first file
        return base_dir / f"{file_prefix}_1{extension}"
    else:
        last_file = files[-1]
        if last_file.stat().st_size < TEN_MB:
            return last_file
        else:
            # need new file
            next_index = len(files) + 1
            return base_dir / f"{file_prefix}_{next_index}{extension}"


def append_json_record(file_path: Path, record: Any) -> None:
    """
    Append a single record to file_path in NDJSON format (one JSON object per line).
    File rollover is handled by get_rollover_file() before we write.
    """
    # Ensure the parent directory exists before appending
    if not file_path.parent.exists():
        file_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created necessary directory: {file_path.parent}")
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    logger.debug(f"Appended record to {file_path}")


def split_and_store_data(enhanced_json: Dict[str, Any]) -> None:
    """
    Splits the validated JSON into separate NDJSON files based on the complete schema structure.
    Each component is stored in its designated directory with appropriate file naming.
    All data is stored in NDJSON format with 10MB file size rollover.
    """
    logger.info("Splitting and storing the enhanced JSON data into NDJSON files...")

    # Store complete history
    history_file = get_rollover_file(BASE_DIR, "all_history", ".ndjson")
    append_json_record(history_file, enhanced_json)

    # Version info
    if "version" in enhanced_json:
        version_file = get_rollover_file(
            BASE_DIR / "metadata", "version_log", ".ndjson"
        )
        append_json_record(version_file, {"version": enhanced_json["version"]})

    # Storage config and retention policies
    if "storage_config" in enhanced_json:
        config_file = get_rollover_file(
            BASE_DIR / "metadata", "storage_config", ".ndjson"
        )
        append_json_record(config_file, enhanced_json["storage_config"])

        # Split retention policies
        if "retention_policy" in enhanced_json["storage_config"]:
            retention_file = get_rollover_file(
                BASE_DIR / "metadata", "retention_policy", ".ndjson"
            )
            append_json_record(
                retention_file, enhanced_json["storage_config"]["retention_policy"]
            )

    # Metadata components
    if "metadata" in enhanced_json:
        # Core metadata
        meta_file = get_rollover_file(BASE_DIR / "metadata", "metadata_log", ".ndjson")
        append_json_record(meta_file, enhanced_json["metadata"])

        # Version history
        if "version_history" in enhanced_json["metadata"]:
            history_file = get_rollover_file(
                BASE_DIR / "metadata", "version_history", ".ndjson"
            )
            append_json_record(
                history_file, enhanced_json["metadata"]["version_history"]
            )

        # Context data
        if "context" in enhanced_json["metadata"]:
            context_file = get_rollover_file(
                BASE_DIR / "metadata", "context_log", ".ndjson"
            )
            append_json_record(context_file, enhanced_json["metadata"]["context"])

            # Environmental factors
            if "environmental_factors" in enhanced_json["metadata"]["context"]:
                env_file = get_rollover_file(
                    BASE_DIR / "metadata", "environmental_factors", ".ndjson"
                )
                append_json_record(
                    env_file,
                    enhanced_json["metadata"]["context"]["environmental_factors"],
                )

        # Identity snapshot components
        if "identity_snapshot" in enhanced_json["metadata"]:
            # Core identity data
            identity_file = get_rollover_file(
                BASE_DIR / "self_model", "identity_snapshot", ".ndjson"
            )
            append_json_record(
                identity_file, enhanced_json["metadata"]["identity_snapshot"]
            )

            # Belief system
            if "belief_system" in enhanced_json["metadata"]["identity_snapshot"]:
                belief_file = get_rollover_file(
                    BASE_DIR / "self_model", "belief_system", ".ndjson"
                )
                append_json_record(
                    belief_file,
                    enhanced_json["metadata"]["identity_snapshot"]["belief_system"],
                )

            # Consciousness matrix
            if "consciousness_matrix" in enhanced_json["metadata"]["identity_snapshot"]:
                consciousness_file = get_rollover_file(
                    BASE_DIR / "consciousness", "consciousness_matrix", ".ndjson"
                )
                append_json_record(
                    consciousness_file,
                    enhanced_json["metadata"]["identity_snapshot"][
                        "consciousness_matrix"
                    ],
                )

            # Adaptive patterns
            if "adaptive_patterns" in enhanced_json["metadata"]["identity_snapshot"]:
                adaptive_file = get_rollover_file(
                    BASE_DIR / "adaptation", "adaptive_patterns", ".ndjson"
                )
                append_json_record(
                    adaptive_file,
                    enhanced_json["metadata"]["identity_snapshot"]["adaptive_patterns"],
                )

    # User input and context
    if "user_input" in enhanced_json:
        input_file = get_rollover_file(
            BASE_DIR / "relationship", "user_input_log", ".ndjson"
        )
        append_json_record(input_file, enhanced_json["user_input"])

        if "context" in enhanced_json["user_input"]:
            user_context_file = get_rollover_file(
                BASE_DIR / "relationship", "user_context", ".ndjson"
            )
            append_json_record(
                user_context_file, enhanced_json["user_input"]["context"]
            )

    # Reflection cycles - split into components
    if "reflection_cycles" in enhanced_json:
        for cycle in enhanced_json["reflection_cycles"]:
            cycle_id = cycle["cycle_id"]

            # Main cycle data
            cycle_file = get_rollover_file(BASE_DIR / "cycles", "cycles_log", ".ndjson")
            append_json_record(cycle_file, cycle)

            # Input processing
            if "input" in cycle:
                input_file = get_rollover_file(
                    BASE_DIR / "cycles", f"input_{cycle_id}", ".ndjson"
                )
                append_json_record(input_file, cycle["input"])

            # Internal processing components
            if "internal_processing" in cycle:
                # Emotional state
                if "emotional_state" in cycle["internal_processing"]:
                    emotional_file = get_rollover_file(
                        BASE_DIR / "emotional", f"emotional_{cycle_id}", ".ndjson"
                    )
                    append_json_record(
                        emotional_file, cycle["internal_processing"]["emotional_state"]
                    )

                    # Emotional memory
                    if (
                        "emotional_memory"
                        in cycle["internal_processing"]["emotional_state"]
                    ):
                        memory_file = get_rollover_file(
                            BASE_DIR / "memory",
                            f"emotional_memory_{cycle_id}",
                            ".ndjson",
                        )
                        append_json_record(
                            memory_file,
                            cycle["internal_processing"]["emotional_state"][
                                "emotional_memory"
                            ],
                        )

                # Cognitive process
                if "cognitive_process" in cycle["internal_processing"]:
                    cognitive_file = get_rollover_file(
                        BASE_DIR / "cognitive", f"cognitive_{cycle_id}", ".ndjson"
                    )
                    append_json_record(
                        cognitive_file,
                        cycle["internal_processing"]["cognitive_process"],
                    )

                    # Metacognition
                    if (
                        "metacognition"
                        in cycle["internal_processing"]["cognitive_process"]
                    ):
                        meta_file = get_rollover_file(
                            BASE_DIR / "cognitive",
                            f"metacognition_{cycle_id}",
                            ".ndjson",
                        )
                        append_json_record(
                            meta_file,
                            cycle["internal_processing"]["cognitive_process"][
                                "metacognition"
                            ],
                        )

                # Consciousness state
                if "consciousness_state" in cycle["internal_processing"]:
                    cons_file = get_rollover_file(
                        BASE_DIR / "consciousness",
                        f"consciousness_{cycle_id}",
                        ".ndjson",
                    )
                    append_json_record(
                        cons_file, cycle["internal_processing"]["consciousness_state"]
                    )

            # Synthesis components
            if "synthesis" in cycle:
                # Key insights and reflections
                insights_file = get_rollover_file(
                    BASE_DIR / "synthesis", f"insights_{cycle_id}", ".ndjson"
                )
                append_json_record(
                    insights_file,
                    {
                        "key_insights": cycle["synthesis"].get("key_insights", []),
                        "emotional_synthesis": cycle["synthesis"].get(
                            "emotional_synthesis", ""
                        ),
                        "meta_reflection": cycle["synthesis"].get(
                            "meta_reflection", ""
                        ),
                    },
                )

                # Emergent consciousness
                if "emergent_consciousness" in cycle["synthesis"]:
                    consciousness_file = get_rollover_file(
                        BASE_DIR / "consciousness", f"emergent_{cycle_id}", ".ndjson"
                    )
                    append_json_record(
                        consciousness_file, cycle["synthesis"]["emergent_consciousness"]
                    )

                # Identity evolution
                if "identity_evolution" in cycle["synthesis"]:
                    evolution_file = get_rollover_file(
                        BASE_DIR / "evolution", f"evolution_{cycle_id}", ".ndjson"
                    )
                    append_json_record(
                        evolution_file, cycle["synthesis"]["identity_evolution"]
                    )

    # Global synthesis components
    if "global_synthesis" in enhanced_json:
        # Overall synthesis
        syn_file = get_rollover_file(BASE_DIR / "synthesis", "synthesis_log", ".ndjson")
        append_json_record(syn_file, enhanced_json["global_synthesis"])

        # Key themes and insights
        themes_file = get_rollover_file(
            BASE_DIR / "synthesis", "themes_and_insights", ".ndjson"
        )
        append_json_record(
            themes_file,
            {
                "key_themes": enhanced_json["global_synthesis"].get("key_themes", []),
                "growth_insights": enhanced_json["global_synthesis"].get(
                    "growth_insights", ""
                ),
                "future_directions": enhanced_json["global_synthesis"].get(
                    "future_directions", []
                ),
            },
        )

        # Consciousness evolution
        if "consciousness_evolution" in enhanced_json["global_synthesis"]:
            cons_evo_file = get_rollover_file(
                BASE_DIR / "consciousness", "global_consciousness", ".ndjson"
            )
            append_json_record(
                cons_evo_file,
                enhanced_json["global_synthesis"]["consciousness_evolution"],
            )

        # Persistent identity
        if "persistent_identity" in enhanced_json["global_synthesis"]:
            identity_file = get_rollover_file(
                BASE_DIR / "self_model", "persistent_identity", ".ndjson"
            )
            append_json_record(
                identity_file, enhanced_json["global_synthesis"]["persistent_identity"]
            )

    # Final output with timestamp
    if "final_output" in enhanced_json:
        output_file = get_rollover_file(
            BASE_DIR / "relationship", "final_output_log", ".ndjson"
        )
        to_store = {
            "timestamp": datetime.now().isoformat(),
            "final_output": enhanced_json["final_output"],
        }
        append_json_record(output_file, to_store)

    logger.info(
        "All enhanced JSON data has been comprehensively split and stored according to schema structure."
    )


def main():
    """Main execution with comprehensive error handling."""
    try:
        # Check for required modules
        if not AVAILABLE_MODULES["json"]:
            logger.error("Required json module not available")
            return

        # Load JSON file
        input_file = Path(BASE_DIR / "output.json")
        if not input_file.exists():
            logger.error(f"Input file not found: {input_file}")
            return

        try:
            with open(input_file, "r", encoding="utf-8") as f:
                json_data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in input file: {e}")
            return
        except Exception as e:
            logger.error(f"Error reading input file: {str(e)}")
            return

        # Validate & enhance JSON
        try:
            result, enhanced_json = validate_and_enhance_json(json_data)
            logger.info(result)
        except Exception as e:
            logger.error(f"Error in validation/enhancement: {str(e)}")
            return

        # Save the enhanced JSON
        try:
            with open(input_file, "w", encoding="utf-8") as f:
                json.dump(enhanced_json, f, indent=2)
            logger.info(f"Enhanced JSON saved to {input_file}")
        except Exception as e:
            logger.error(f"Error saving enhanced JSON: {str(e)}")
            return

        # Split & store data
        try:
            split_and_store_data(enhanced_json)
        except Exception as e:
            logger.error(f"Error splitting/storing data: {str(e)}")
            return

    except Exception as e:
        logger.error(f"Unexpected error in main: {str(e)}")
        raise


if __name__ == "__main__":
    main()
