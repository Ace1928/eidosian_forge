import os
import json
from typing import Dict, Any, Optional, Literal, Mapping, Union, List
from dataclasses import dataclass, field, asdict
import logging
import logging.config
from dotenv import load_dotenv
import importlib
import sys
import uuid
from logging import Formatter, StreamHandler

# Load environment variables from .env file
load_dotenv()

# ⚙️ Core Configuration - Environment Variables and Defaults
DEFAULT_DEVELOPMENT_DIR = "/Development"
EIDOS_PROJECT_DIR_ENV = "EIDOS_PROJECT_DIR"
DEFAULT_EIDOS_PROJECT_DIR = os.path.join(DEFAULT_DEVELOPMENT_DIR, "eidos_project")
EIDOS_PROJECT_DIR = os.environ.get(EIDOS_PROJECT_DIR_ENV, DEFAULT_EIDOS_PROJECT_DIR)

LOGGING_CONFIG_DIR_ENV = "LOGGING_CONFIG_DIR"
DEFAULT_LOGGING_CONFIG_DIR = os.path.join(EIDOS_PROJECT_DIR, "config")
LOGGING_CONFIG_DIR = os.environ.get(LOGGING_CONFIG_DIR_ENV, DEFAULT_LOGGING_CONFIG_DIR)

UNIVERSAL_CONFIG_FILE_ENV = "UNIVERSAL_CONFIG_FILE"
DEFAULT_UNIVERSAL_CONFIG_FILE = "universal_config.json"
UNIVERSAL_CONFIG_PATH = os.path.join(
    EIDOS_PROJECT_DIR,
    os.environ.get(UNIVERSAL_CONFIG_FILE_ENV, DEFAULT_UNIVERSAL_CONFIG_FILE),
)

UNIVERSAL_LOGGING_CONFIG_FILE_ENV = "UNIVERSAL_LOGGING_CONFIG_FILE"
DEFAULT_UNIVERSAL_LOGGING_CONFIG_FILE = "universal_logging_config.json"
UNIVERSAL_LOGGING_CONFIG_PATH = os.path.join(
    LOGGING_CONFIG_DIR,
    os.environ.get(
        UNIVERSAL_LOGGING_CONFIG_FILE_ENV, DEFAULT_UNIVERSAL_LOGGING_CONFIG_FILE
    ),
)

DEFAULT_LOG_LEVEL_ENV = "DEFAULT_LOG_LEVEL"
DEFAULT_LOG_LEVEL_STR = os.environ.get(DEFAULT_LOG_LEVEL_ENV, "INFO").upper()
DEFAULT_LOG_LEVEL = getattr(logging, DEFAULT_LOG_LEVEL_STR, logging.INFO)

DEFAULT_LOG_FORMAT_TEXT_ENV = "DEFAULT_LOG_FORMAT_TEXT"
DEFAULT_LOG_FORMAT_TEXT = os.environ.get(
    DEFAULT_LOG_FORMAT_TEXT_ENV,
    "%(asctime)s - %(levelname)s - [%(name)s] - %(filename)s:%(lineno)d - %(message)s",
)

DEFAULT_LOG_FORMAT_JSON_ENV = "DEFAULT_LOG_FORMAT_JSON"
DEFAULT_LOG_FORMAT_JSON = os.environ.get(
    DEFAULT_LOG_FORMAT_JSON_ENV,
    '{"time": "%(asctime)s", "level": "%(levelname)s", "name": "%(name)s", "module": "%(module)s", '
    '"function": "%(funcName)s", "line": "%(lineno)d", "message": "%(message)s"}',
)


def ensure_config_dir_exists() -> None:
    """Ensures the universal configuration directory exists."""
    os.makedirs(os.path.dirname(UNIVERSAL_CONFIG_PATH), exist_ok=True)


def load_config_from_disk() -> Dict[str, Any]:
    """Loads universal configuration from disk or returns defaults. 💾"""
    ensure_config_dir_exists()
    try:
        if os.path.exists(UNIVERSAL_CONFIG_PATH):
            with open(UNIVERSAL_CONFIG_PATH, "r", encoding="utf-8") as config_file:
                return json.load(config_file)
    except (json.JSONDecodeError, OSError) as e:
        logging.error(f"Error loading config from {UNIVERSAL_CONFIG_PATH}: {e}")
    return {}


def save_config_to_disk(config: Dict[str, Any]) -> None:
    """Saves the universal configuration to disk. 💾"""
    ensure_config_dir_exists()
    try:
        with open(UNIVERSAL_CONFIG_PATH, "w", encoding="utf-8") as config_file:
            json.dump(config, config_file, indent=4)
    except OSError as e:
        logging.error(f"Error saving config to {UNIVERSAL_CONFIG_PATH}: {e}")


@dataclass
class UniversalConfig:
    """✨ Unified configuration dataclass for UniversalUtility. ⚙️

    Combines configurations for profiling, resource monitoring, and log formatting,
    embodying Eidosian principles of comprehensive configurability. 🚀
    """

    # Profiler Configuration
    profiler_enabled: bool = field(
        default=True, metadata={"description": "Enable or disable the profiler."}
    )

    # Monitor Configuration
    monitor_enabled: bool = field(
        default=False,
        metadata={"description": "Enable or disable the resource monitor."},
    )
    monitor_interval: float = field(
        default=5.0,
        metadata={"description": "Interval for resource monitoring in seconds."},
    )

    # Formatter Configuration
    formatter_enabled: bool = field(
        default=False,
        metadata={"description": "Enable or disable the custom formatter."},
    )
    fmt: Optional[str] = field(
        default=None, metadata={"description": "Format string for log message."}
    )
    datefmt: Optional[str] = field(
        default=None, metadata={"description": "Format string for the date."}
    )
    style: Literal["%", "{", "$"] = field(
        default="%", metadata={"description": "Style of the format string."}
    )
    prefix: str = field(
        default="✨ ", metadata={"description": "Prefix added to log message."}
    )
    suffix: str = field(
        default=" ✨", metadata={"description": "Suffix added to log message."}
    )
    use_rich: bool = field(
        default=False, metadata={"description": "Enable rich text formatting."}
    )
    rich_console_width: Optional[int] = field(
        default=None, metadata={"description": "Width of the rich console."}
    )
    rich_force_terminal: Optional[bool] = field(
        default=None,
        metadata={"description": "Force rich output even if not a terminal."},
    )
    rich_color_system: Literal["standard", "256", "truecolor", "no"] = field(
        default="standard", metadata={"description": "Color system for rich output."}
    )
    rich_color_map: Dict[str, Optional[str]] = field(
        default_factory=lambda: {
            "standard": "standard",
            "256": "256",
            "truecolor": "truecolor",
            "no": None,
        },
        metadata={"description": "Color map for rich output."},
    )

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "UniversalConfig":
        """Creates a UniversalConfig object from a dictionary, handling missing or extra keys."""
        default_config = asdict(cls())
        merged_config = cls._recursive_merge(default_config, config_dict)
        return cls(**merged_config)

    @classmethod
    def _recursive_merge(
        cls, default: Dict[str, Any], config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Recursively merges a configuration dictionary with default values, handling type validation."""
        merged = default.copy()
        for key, value in config.items():
            field_type = cls.__dataclass_fields__.get(key)
            if field_type:
                if isinstance(value, dict) and isinstance(
                    default.get(key), dict
                ):  # Recursive merge for nested dicts
                    merged[key] = cls._recursive_merge(default[key], value)
                elif field_type.type is bool and not isinstance(value, bool):
                    merged[key] = str(value).lower() == "true"
                elif field_type.type is int and not isinstance(value, int):
                    if isinstance(value, (int, float)):
                        try:
                            merged[key] = int(float(value))
                        except (ValueError, TypeError):
                            logging.warning(
                                f"Invalid value for {key}: {value}, using default."
                            )
                            merged[key] = default.get(key)
                    else:
                        logging.warning(
                            f"Invalid value for {key}: {value}, using default."
                        )
                        merged[key] = default.get(key)
                elif field_type.type is float and not isinstance(value, float):
                    if value is not None:
                        if isinstance(value, (int, float)):
                            try:
                                merged[key] = float(value)
                            except (ValueError, TypeError):
                                logging.warning(
                                    f"Invalid value for {key}: {value}, using default."
                                )
                                merged[key] = default.get(key)
                        else:
                            logging.warning(
                                f"Invalid value for {key}: {value}, using default."
                            )
                            merged[key] = default.get(key)
                    else:
                        logging.warning(
                            f"Invalid value for {key}: {value}, using default."
                        )
                        merged[key] = default.get(key)
                elif field_type.type is str and not isinstance(value, str):
                    merged[key] = str(value)
                elif (
                    field_type.type == Optional[str]
                    and value is not None
                    and not isinstance(value, str)
                ):
                    merged[key] = str(value)
                elif field_type.type == Literal["%", "{", "$"] and value not in [
                    "%",
                    "{",
                    "$",
                ]:
                    logging.warning(f"Invalid value for {key}: {value}, using default.")
                    merged[key] = default.get(key)
                elif field_type.type == Literal[
                    "standard", "256", "truecolor", "no"
                ] and value not in ["standard", "256", "truecolor", "no"]:
                    logging.warning(f"Invalid value for {key}: {value}, using default.")
                    merged[key] = default.get(key)
                elif field_type.type == Dict[str, Optional[str]] and not isinstance(
                    value, dict
                ):
                    logging.warning(f"Invalid value for {key}: {value}, using default.")
                    merged[key] = default.get(key)
                elif (
                    field_type.type == Optional[int]
                    and value is not None
                    and not isinstance(value, int)
                ):
                    if isinstance(value, (int, float)):
                        try:
                            merged[key] = int(float(value))
                        except (ValueError, TypeError):
                            logging.warning(
                                f"Invalid value for {key}: {value}, using default."
                            )
                            merged[key] = default.get(key)
                    else:
                        logging.warning(
                            f"Invalid value for {key}: {value}, using default."
                        )
                        merged[key] = default.get(key)
                elif (
                    field_type.type == Optional[bool]
                    and value is not None
                    and not isinstance(value, bool)
                ):
                    merged[key] = str(value).lower() == "true"
                elif isinstance(field_type.type, type) and isinstance(
                    value, field_type.type
                ):
                    merged[key] = value
                elif isinstance(field_type.type, type) and not isinstance(
                    value, field_type.type
                ):
                    logging.warning(f"Invalid type for {key}: {value}, using default.")
                    merged[key] = default.get(key)
                elif isinstance(field_type.type, type) and isinstance(value, list):
                    merged[key] = value
                else:
                    logging.warning(f"Skipping unknown type for {key}: {value}")
                    merged[key] = value
            else:
                logging.warning(f"Skipping unknown key: {key}")
        return merged


# Logging Configuration
LOG_LEVEL = DEFAULT_LOG_LEVEL
LOG_FORMAT = DEFAULT_LOG_FORMAT_TEXT
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class EidosFormatter(Formatter):
    def __init__(
        self,
        format_string: str,
        datefmt: Optional[str],
        use_json: bool,
        include_uuid: bool,
        **kwargs,
    ):
        super().__init__(format_string, datefmt)
        self.use_json = use_json
        self.include_uuid = include_uuid

    def format(self, record: logging.LogRecord) -> str:
        if self.include_uuid:
            setattr(record, "uuid", uuid.uuid4())
        log_record = {
            "time": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "filename": record.filename,
            "lineno": record.lineno,
            "module": record.module,
            "function": record.funcName,
            "message": record.getMessage(),
        }

        if self.use_json:
            return json.dumps(log_record)
        else:
            return super().format(record)


def configure_logging(
    log_config_path: str = UNIVERSAL_LOGGING_CONFIG_PATH, attempt=1
) -> None:
    """Configures logging using a JSON file or defaults."""
    if os.path.exists(log_config_path):
        try:
            with open(log_config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                # Dynamically import the formatter
                if "formatters" in config:
                    keys_to_delete = []
                    for formatter_name, formatter_config in config[
                        "formatters"
                    ].items():
                        if "()" in formatter_config:
                            if (
                                isinstance(formatter_config["()"], str)
                                and formatter_config["()"] == "utility.EidosFormatter"
                            ):
                                formatter_config["()"] = EidosFormatter
                            elif formatter_config["()"] == EidosFormatter:
                                continue
                            else:
                                logging.error(
                                    f"Error importing formatter {formatter_config['()']}, removing formatter."
                                )
                                keys_to_delete.append(formatter_name)
                                continue
                    for key in keys_to_delete:
                        del config["formatters"][key]
                    if keys_to_delete:
                        logging.warning(
                            f"Removed invalid formatters, reconfiguring logging. Attempt: {attempt}"
                        )
                        save_config_to_disk(config)
                        if attempt < 3:
                            configure_logging(log_config_path, attempt + 1)
                            return
                        else:
                            logging.error(
                                f"Failed to configure logging after multiple attempts. Falling back to basic configuration."
                            )
                            logging.basicConfig(
                                level=LOG_LEVEL,
                                format=LOG_FORMAT,
                                datefmt=LOG_DATE_FORMAT,
                            )
                            return
                logging.config.dictConfig(config)
                logging.info(f"Logging configured from {log_config_path}")
                return
        except (json.JSONDecodeError, KeyError, TypeError, OSError) as e:
            logging.error(f"Error loading logging config from {log_config_path}: {e}")
    # Fallback to basic configuration
    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    logging.warning(f"Using default logging configuration.")


configure_logging()
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    # Example usage
    config = load_config_from_disk()
    print("Loaded config:", config)

    config["formatter"] = {
        "enabled": True,
        "fmt": "%(message)s",
        "style": "$",
        "rich_color_map": {"standard": "red"},
    }
    save_config_to_disk(config)
    print("Config updated and saved.")

    loaded_config = load_config_from_disk()
    print("Loaded config after update:", loaded_config)

    universal_config = UniversalConfig.from_dict(loaded_config)
    print("UniversalConfig object:", universal_config)
    print("UniversalConfig as dict:", asdict(universal_config))
