from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required, get_jwt_identity
from typing import Dict, Any, Tuple, Union

    Endpoint to manage user-specific settings. Handles both GET and POST requests for fetching and updating settings, respectively.
    Returns:
        APIResponse: A tuple containing a Flask Response object with a JSON message and an appropriate status code.
    