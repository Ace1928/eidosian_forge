from flask import Blueprint, jsonify, request
from werkzeug.security import generate_password_hash, check_password_hash
from flask_jwt_extended import create_access_token
from typing import Dict, Any, Tuple

    Endpoint for user login.

    Returns:
    - AuthResponse: A tuple containing a Flask Response object with a JSON message containing the JWT access token or an error message, and an appropriate status code.
    