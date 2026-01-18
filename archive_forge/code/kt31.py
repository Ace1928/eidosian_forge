from flask import Blueprint, jsonify, request, Response
from werkzeug.security import generate_password_hash, check_password_hash
from flask_jwt_extended import create_access_token, JWTManager
from extensions import jwt
from typing import Any, Dict, Tuple

auth_bp = Blueprint("auth", __name__)

# Type alias for clarity and to ensure type safety throughout the module
UserData = Dict[str, Any]
ResponseData = Dict[str, Any]
AuthResponse = Tuple[ResponseData, int]


def register_user(user_data: UserData) -> ResponseData:
    """
    Registers a new user by hashing their password and storing user data.

    :param user_data: A dictionary containing the username and password of the user.
    :return: A dictionary with a message indicating successful registration.
    """
    username: str = user_data["username"]
    password: str = user_data["password"]
    hashed_password: str = generate_password_hash(password)
    # Here, you would store the username and hashed_password in your database.
    # For demonstration, we're just returning a success message.
    return {"msg": f"User {username} registered successfully"}


@auth_bp.route("/register", methods=["POST"])
def register() -> Response:
    """
    Endpoint for user registration.

    :return: A Flask Response object with a JSON message.
    """
    user_data: UserData = request.get_json()
    result: ResponseData = register_user(user_data)
    return jsonify(result), 201


def authenticate_user(username: str, password: str) -> ResponseData:
    """
    Authenticates a user by checking their password against the hashed version.

    :param username: The username of the user.
    :param password: The password provided by the user for authentication.
    :return: A dictionary containing the JWT access token for the authenticated user.
    """
    # Here, you would retrieve the user's hashed password from your database.
    # For demonstration, we're assuming authentication is successful.
    access_token: str = create_access_token(identity=username)
    return {"access_token": access_token}


@auth_bp.route("/login", methods=["POST"])
def login() -> Response:
    """
    Endpoint for user login.

    :return: A Flask Response object with a JSON message containing the JWT access token.
    """
    login_data: UserData = request.get_json()
    username: str = login_data["username"]
    password: str = login_data["password"]
    result: ResponseData = authenticate_user(username, password)
    return jsonify(result), 200
