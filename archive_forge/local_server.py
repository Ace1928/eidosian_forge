"""
Local server implementation for ace_tools.py

Provides a robust local HTTP server that handles requests when the main server is unavailable.
Implements all core functionality with local storage fallbacks.
"""


import os
import json
import logging
import threading
import time
import socket
from http.server import HTTPServer, BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Any, Optional, Union
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import io
import base64
import traceback
from contextlib import contextmanager
from http import client

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Server configuration with fallbacks and validation
HOST = os.getenv("LOCAL_SERVER_HOST", "localhost") 
try:
    PORT = int(os.getenv("LOCAL_SERVER_PORT", "8080"))
except ValueError:
    logger.warning("Invalid port specified, using default 8080")
    PORT = 8080

STORAGE_DIR = Path(os.getenv("LOCAL_STORAGE_DIR", ".ace_tools"))
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# Configurable timeouts
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
SHUTDOWN_TIMEOUT = int(os.getenv("SHUTDOWN_TIMEOUT", "5"))

class RobustRequestHandler(BaseHTTPRequestHandler):
    """Enhanced request handler with robust error handling and recovery"""
    
    timeout = REQUEST_TIMEOUT
    protocol_version = 'HTTP/1.1'  # Explicitly set HTTP version
    
    @contextmanager
    def error_boundary(self, error_msg: str):
        """Context manager for consistent error handling"""
        try:
            yield
        except (ConnectionError, client.RemoteDisconnected) as e:
            logger.error(f"Connection error in {error_msg}: {e}")
            # Don't try to send response for connection errors
            return
        except Exception as e:
            logger.error(f"{error_msg}: {e}\n{traceback.format_exc()}")
            self._send_response({"error": str(e)}, 500)

    def _send_response(self, data: Dict[str, Any], status: int = 200) -> None:
        """Enhanced response sender with retry logic and connection handling"""
        max_retries = 3
        response_json = json.dumps(data).encode()
        
        for attempt in range(max_retries):
            try:
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(response_json)))
                self.send_header('Connection', 'keep-alive')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                # Send in chunks to avoid buffer issues
                chunk_size = 8192
                for i in range(0, len(response_json), chunk_size):
                    chunk = response_json[i:i + chunk_size]
                    try:
                        self.wfile.write(chunk)
                        self.wfile.flush()
                    except (ConnectionError, client.RemoteDisconnected):
                        logger.warning("Client disconnected during response")
                        return
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                logger.warning(f"Retry {attempt + 1}/{max_retries} sending response: {e}")
                time.sleep(0.5)

    def _safely_read_file(self, path: Union[str, Path]) -> pd.DataFrame:
        """Safely read files with validation"""
        try:
            if isinstance(path, str):
                path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            if path.suffix.lower() == '.csv':
                return pd.read_csv(path)
            raise ValueError(f"Unsupported file type: {path.suffix}")
        except Exception as e:
            logger.error(f"Error reading file {path}: {e}")
            raise

    def _safely_write_file(self, path: Path, data: Any) -> None:
        """Safely write files with backup"""
        backup_path = path.with_suffix(path.suffix + '.bak')
        if path.exists():
            path.rename(backup_path)
        try:
            if isinstance(data, pd.DataFrame):
                data.to_csv(path, index=False)
            elif isinstance(data, dict):
                with open(path, 'w') as f:
                    json.dump(data, f, indent=2)
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")
        except Exception as e:
            if backup_path.exists():
                backup_path.rename(path)
            raise

    def _handle_display_dataframe(self, data: Dict[str, Any]) -> None:
        """Enhanced dataframe handler with validation"""
        with self.error_boundary("Error handling dataframe display"):
            path = data.get("path")
            title = data.get("title", "default_title")  # Provide a default title if not present
            
            if not path:
                raise ValueError("Path is required and cannot be None")
            
            df = self._safely_read_file(path)
            local_path = STORAGE_DIR / f"{title}.csv"
            self._safely_write_file(local_path, df)
            
            self._send_response({"value": None})

    def _handle_display_chart(self, data: Dict[str, Any]) -> None:
        """Enhanced chart handler with metadata validation"""
        with self.error_boundary("Error handling chart display"):
            chart_data = {
                "path": data.get("path"),
                "title": data.get("title", "Untitled"),
                "type": data.get("chart_type", "unknown"),
                "metadata": data.get("metadata", {}),
                "timestamp": time.time()
            }
            
            chart_path = STORAGE_DIR / f"chart_{chart_data['title']}_{int(time.time())}.json"
            self._safely_write_file(chart_path, chart_data)
            self._send_response({"value": None})

    def _handle_matplotlib_fallback(self, data: Dict[str, Any]) -> None:
        """Enhanced matplotlib fallback handler"""
        with self.error_boundary("Error handling matplotlib fallback"):
            fallback_data = {
                "reason": data.get("reason", "Unknown"),
                "metadata": data.get("metadata", {}),
                "timestamp": time.time()
            }
            
            fallback_id = data.get("metadata", {}).get("id", f"fallback_{int(time.time())}")
            fallback_path = STORAGE_DIR / f"matplotlib_{fallback_id}.json"
            
            self._safely_write_file(fallback_path, fallback_data)
            logger.info(f"Matplotlib fallback: {fallback_data['reason']}")
            self._send_response({"value": None})

    def _handle_exception_logging(self, data: Dict[str, Any]) -> None:
        """Enhanced exception handler with detailed logging"""
        with self.error_boundary("Error handling exception logging"):
            exc_data = {
                **data,
                "timestamp": time.time(),
                "handler_info": {
                    "client_address": self.client_address[0],
                    "command": self.command,
                    "path": self.path
                }
            }
            
            exc_id = data.get("exception", {}).get("id", f"error_{int(time.time())}")
            error_path = STORAGE_DIR / f"error_{exc_id}.json"
            
            self._safely_write_file(error_path, exc_data)
            logger.error(f"Exception logged: {data.get('message')}")
            self._send_response({"value": None})

    def do_POST(self):
        """Enhanced POST handler with request validation"""
        with self.error_boundary("Error processing POST request"):
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length > 10 * 1024 * 1024:  # 10MB limit
                raise ValueError("Request too large")
                
            post_data = json.loads(self.rfile.read(content_length))
            
            handlers = {
                "/ace_tools/call_function": {
                    "display_dataframe_to_user": self._handle_display_dataframe,
                    "display_chart_to_user": self._handle_display_chart,
                    "log_matplotlib_img_fallback": self._handle_matplotlib_fallback,
                    "display_matplotlib_image_to_user": self._handle_matplotlib_fallback
                },
                "/ace_tools/log_exception": lambda data: self._handle_exception_logging(data)
            }
            
            if self.path in handlers:
                if self.path == "/ace_tools/call_function":
                    method = post_data.get("method")
                    kwargs = post_data.get("kwargs", {})
                    if method in handlers[self.path]:
                        handlers[self.path][method](kwargs)
                    else:
                        self._send_response({"value": None})
                else:
                    handlers[self.path](post_data)
            else:
                self._send_response({"error": "Invalid endpoint"}, 404)

class RobustThreadingHTTPServer(ThreadingHTTPServer):
    """Enhanced HTTP server with connection management"""
    def server_bind(self):
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        # Enable TCP Keepalive
        if hasattr(socket, "TCP_KEEPIDLE"):
            self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 60)
        if hasattr(socket, "TCP_KEEPINTVL"):
            self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10)
        if hasattr(socket, "TCP_KEEPCNT"):
            self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 5)
        super().server_bind()

def run_server(host: str = HOST, port: int = PORT) -> HTTPServer:
    """Enhanced server runner with health checks"""
    try:
        server = RobustThreadingHTTPServer((host, port), RobustRequestHandler)
        logger.info(f"Starting local server on {host}:{port}")
        
        def health_check():
            while True:
                try:
                    with socket.create_connection((host, port), timeout=5):
                        pass
                except Exception as e:
                    logger.error(f"Server health check failed: {e}")
                time.sleep(30)
        
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        health_thread = threading.Thread(target=health_check, daemon=True)
        
        server_thread.start()
        health_thread.start()
        
        return server
        
    except Exception as e:
        logger.error(f"Failed to start local server: {e}\n{traceback.format_exc()}")
        raise

def safely_write_file(path: Path, data: Any) -> None:
    """Safely write files with backup"""
    backup_path = path.with_suffix(path.suffix + '.bak')
    if path.exists():
        path.rename(backup_path)
    try:
        if isinstance(data, pd.DataFrame):
            data.to_csv(path, index=False)
        elif isinstance(data, dict):
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
    except Exception as e:
        if backup_path.exists():
            backup_path.rename(path)
        raise

def main():
    """Main function to validate and test all server functionality"""
    logger.info("Starting comprehensive server validation and testing")
    
    # Test configuration and environment
    logger.info("Testing configuration...")
    assert isinstance(PORT, int), "Port must be an integer"
    assert 0 < PORT < 65536, "Port must be in valid range"
    assert STORAGE_DIR.exists(), "Storage directory must exist"
    assert STORAGE_DIR.is_dir(), "Storage path must be a directory"
    
    # Test file operations
    logger.info("Testing file operations...")
    test_df = pd.DataFrame({'test': [1,2,3]})
    test_file = STORAGE_DIR / "test.csv"
    safely_write_file(test_file, test_df)
    assert test_file.exists(), "File write failed"
    read_df = pd.read_csv(test_file)
    assert read_df.equals(test_df), "File read/write integrity check failed"
    
    # Test server initialization
    logger.info("Testing server initialization...")
    server = run_server()
    time.sleep(1)  # Allow server to start
    
    # Keep server running if started by another program
    if not os.environ.get("LOCAL_SERVER_TEST_MODE"):
        try:
            while True:
                time.sleep(60)  # Keep alive
        except KeyboardInterrupt:
            logger.info("Server shutdown requested")
            server.shutdown()
            server.server_close()
    else:
        # Test connection
        logger.info("Testing server connection...")
        try:
            with socket.create_connection((HOST, PORT), timeout=5):
                logger.info("Server connection successful")
        except Exception as e:
            logger.error(f"Server connection failed: {e}")
            raise
            
        # Clean up test artifacts
        logger.info("Cleaning up test artifacts...")
        test_file.unlink()
        
        # Shutdown server
        logger.info("Testing server shutdown...")
        server.shutdown()
        server.server_close()
        
        logger.info("All validation tests completed successfully")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Validation failed: {e}\n{traceback.format_exc()}")
        raise
