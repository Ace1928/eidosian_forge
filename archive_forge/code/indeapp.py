"""
Module: app.py

This module initializes the Flask application and integrates all components of the trading bot application, including authentication, API endpoints, configuration, and extensions.

Dependencies:
- Flask: A web framework for building the API endpoints.
- Flask-JWT-Extended: For handling JSON Web Tokens (JWTs) for authentication.
- Flask-SocketIO: For real-time communication between the client and server.
- Werkzeug: Provides utilities for secure password hashing.
- os: To access environment variables for configuration.

Classes:
- None

Functions:
- create_app(config_class=Config) -> Flask:
    Initializes and configures the Flask application.

Variables:
- app: Instance of the Flask application.

Authorship and Versioning Details:
    Author: Lloyd Handyside
    Creation Date: 2024-04-16 (ISO 8601 Format)
    Last Modified: 2024-04-16 (ISO 8601 Format)
    Version: 1.0.0 (Semantic Versioning)
    Contact: lloyd.handyside@neuroforge.io
    Ownership: Neuro Forge
    Status: Draft (Subject to change)
"""

from flask import Flask
from flask_jwt_extended import JWTManager
from flask_socketio import SocketIO
import os
from typing import Type

from scripts.trading_bot.indeplugins import jwt, socketio
from scripts.trading_bot.indeauth import auth_bp
from scripts.trading_bot.indeapi import api_bp
from scripts.trading_bot.indeconfig import Config


def create_app(config_class: Type[Config]) -> Flask:
    """
    Application factory that initializes and configures the Flask application.

    Parameters:
    - config_class: Type[Config] - The configuration class to use for application settings.

    Returns:
    - Flask: The initialized Flask application.
    """
    app = Flask(__name__)
    app.config.from_object(config_class)

    jwt.init_app(app)
    socketio.init_app(app, cors_allowed_origins="*")

    app.register_blueprint(auth_bp, url_prefix="/api")
    app.register_blueprint(api_bp, url_prefix="/api")

    return app


if __name__ == "__main__":
    app = create_app(Config)
    socketio.run(app, debug=True)
