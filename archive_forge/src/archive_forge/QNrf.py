from flask import Blueprint, jsonify, request
from werkzeug.security import generate_password_hash, check_password_hash
from flask_jwt_extended import create_access_token
from extensions import jwt

auth_bp = Blueprint("auth", __name__)


@auth_bp.route("/register", methods=["POST"])
def register():
    # Implement registration logic
    # This should interact with your users_db or a proper database
    pass


@auth_bp.route("/login", methods=["POST"])
def login():
    # Implement login logic
    # This should verify credentials and return a JWT token
    pass
