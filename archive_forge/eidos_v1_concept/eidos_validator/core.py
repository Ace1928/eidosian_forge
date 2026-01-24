"""
Eidos Validator Core
-----------------
Core integration for the Eidos Validator system, providing a unified API.

This module serves as the primary entry point for the validator system,
orchestrating the various components together while maintaining separation
of concerns.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union, List, cast

from .cache import CacheManager
from .validator import SchemaValidator
from .enhancer import JsonEnhancer
from .storage import StorageManager
from .system import SystemInfoCollector
from .config import BASE_DIR, REQUIRED_DIRS, REQUIRED_MODULES
from .utils import check_module_availability, ensure_directories

# Get logger
logger = logging.getLogger('eidos_validator.core')

class EidosValidator:
    """
    Core validator class that orchestrates all components of the Eidos validator.
    
    This class serves as the main interface for validating, enhancing, and
    storing Eidosian data structures. It coordinates the various specialized
    components while maintaining separation of concerns.
    """
    
    def __init__(
        self, 
        schema_path: Optional[Union[str, Path]] = None,
        base_dir: Optional[Union[str, Path]] = None,
        cache_dir: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Initialize the Eidos validator with configurable paths.
        
        Args:
            schema_path: Path to the JSON schema file (default: BASE_DIR / 'eidosian_io_schema.json')
            base_dir: Base directory for file operations (default: config.BASE_DIR)
            cache_dir: Directory for cache files (default: config.CACHE_DIR)
            
        Raises:
            FileNotFoundError: If schema_path is provided but not found
        """
        # Initialize the base directory
        self.base_dir = Path(base_dir) if base_dir else BASE_DIR
        
        # Setup cache directory and manager
        self.cache_dir = Path(cache_dir) if cache_dir else self.base_dir / '.cache'
        self.cache_manager = CacheManager(self.cache_dir)
        
        # Check which modules are available
        self.available_modules = check_module_availability(REQUIRED_MODULES)
        
        # Ensure required directories exist
        ensure_directories(self.base_dir, REQUIRED_DIRS)
        
        # Initialize components
        schema_path = schema_path or self.base_dir / 'eidosian_io_schema.json'
        self.validator = SchemaValidator(schema_path, self.cache_manager)
        self.validator.set_available_modules(self.available_modules)
        
        # System info collector with appropriate dependency injection
        self.system_info = SystemInfoCollector(self.cache_manager)
        self.system_info.set_available_modules(self.available_modules)
        
        # JSON enhancer with system_info dependency
        self.enhancer = JsonEnhancer(self.system_info)
        self.enhancer.set_available_modules(self.available_modules)
        
        # Storage manager for file operations
        self.storage = StorageManager(self.base_dir)
        
        logger.info(f"Eidos Validator initialized with base directory: {self.base_dir}")

    def process_json(
        self, 
        json_data: Dict[str, Any], 
        validate: bool = True,
        enhance: bool = True,
        store: bool = True
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Process JSON data through validation, enhancement, and storage.
        
        This method orchestrates the complete flow of JSON processing:
        1. Validates against schema (if validate=True)
        2. Enhances with system info and defaults (if enhance=True)
        3. Stores to appropriate files (if store=True)
        
        Args:
            json_data: The input JSON data to process
            validate: Whether to validate against schema
            enhance: Whether to enhance the JSON
            store: Whether to store the processed JSON
            
        Returns:
            Tuple[bool, str, Dict[str, Any]]: (success, message, processed_json)
            
        Raises:
            ValueError: If json_data is not a dictionary
        """
        if not isinstance(json_data, dict):
            raise ValueError("Input must be a dictionary (parsed JSON)")
        
        success = True
        message = "Processing completed successfully"
        processed_json = json_data.copy()
        
        logger.info("Beginning JSON processing pipeline")
        
        # Step 1: Validate
        if validate:
            logger.debug("Validating JSON against schema")
            is_valid, validation_message = self.validator.validate(processed_json)
            if not is_valid:
                success = False
                message = validation_message
                logger.warning(f"Validation failed: {validation_message}")
            else:
                logger.info("Validation successful")

        # Step 2: Enhance
        if enhance:
            logger.debug("Enhancing JSON with system info and defaults")
            try:
                processed_json = self.enhancer.enhance_json(processed_json)
                logger.info("Enhancement successful")
            except Exception as e:
                success = False
                message = f"Enhancement failed: {str(e)}"
                logger.error(f"Error during enhancement: {str(e)}", exc_info=True)
        
        # Step 3: Store
        if store and success:
            logger.debug("Storing processed JSON")
            try:
                self.storage.store_json(processed_json)
                logger.info("Storage successful")
            except Exception as e:
                success = False
                message = f"Storage failed: {str(e)}"
                logger.error(f"Error during storage: {str(e)}", exc_info=True)

        logger.info(f"JSON processing complete: {message}")
        return success, message, processed_json
    
    def process_file(
        self, 
        file_path: Union[str, Path],
        validate: bool = True,
        enhance: bool = True, 
        store: bool = True,
        output_path: Optional[Union[str, Path]] = None
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Process a JSON file through validation, enhancement, and storage.
        
        Args:
            file_path: Path to the JSON file to process
            validate: Whether to validate against schema
            enhance: Whether to enhance the JSON
            store: Whether to store the processed JSON
            output_path: Optional path to save the processed JSON
            
        Returns:
            Tuple[bool, str, Dict[str, Any]]: (success, message, processed_json)
            
        Raises:
            FileNotFoundError: If file_path does not exist
            json.JSONDecodeError: If file contains invalid JSON
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        logger.info(f"Processing file: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                
            success, message, processed_json = self.process_json(
                json_data, 
                validate=validate,
                enhance=enhance,
                store=store
            )
            
            # Save processed JSON if output path provided
            if output_path:
                output_path = Path(output_path)
                # Create parent directories if needed
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(processed_json, f, indent=2)
                logger.info(f"Saved processed JSON to {output_path}")
                
            return success, message, processed_json
            
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON in file {file_path}: {e}"
            logger.error(error_msg)
            raise
        except Exception as e:
            error_msg = f"Error processing file {file_path}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get comprehensive system information.
        
        Returns:
            Dict[str, Any]: Dictionary of system information
        """
        return self.system_info.get_system_info()
    
    def clear_caches(self) -> int:
        """
        Clear all caches.
        
        Returns:
            int: Number of cache entries cleared
        """
        return self.cache_manager.clear()
