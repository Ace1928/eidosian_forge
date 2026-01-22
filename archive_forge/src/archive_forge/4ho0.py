from flask import Flask, jsonify, request
from flask_jwt_extended import (
    JWTManager,
    create_access_token,
    jwt_required,
    get_jwt_identity,
)
from flask_socketio import SocketIO
from werkzeug.security import generate_password_hash, check_password_hash
import os

from flask import Flask
from extensions import jwt, socketio
from bp_auth import auth_bp
from bp_api import api_bp
from config import Config


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    jwt.init_app(app)
    socketio.init_app(app, cors_allowed_origins="*")

    app.register_blueprint(auth_bp, url_prefix="/api")
    app.register_blueprint(api_bp, url_prefix="/api")

    return app


app.config["JWT_SECRET_KEY"] = os.environ.get("JWT_SECRET_KEY", "your_jwt_secret_key")
app.config["JWT_SECRET_KEY"] = os.environ.get("JWT_SECRET_KEY", "your_jwt_secret_key")
jwt = JWTManager(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Dummy database simulation
users_db = {"admin": generate_password_hash("password123")}
strategies_db = {}
performance_db = {}
settings_db = {}


@app.route("/api/register", methods=["POST"])
def register():
    username = request.json.get("username", None)
    password = request.json.get("password", None)
    if not username or not password:
        return jsonify({"msg": "Missing username or password"}), 400
    if username in users_db:
        return jsonify({"msg": "Username already exists"}), 409
    users_db[username] = generate_password_hash(password)
    return jsonify({"msg": "User registered successfully"}), 201


@app.route("/api/login", methods=["POST"])
def login():
    username = request.json.get("username", None)
    password = request.json.get("password", None)
    if not username or not password:
        return jsonify({"msg": "Missing username or password"}), 400
    if username not in users_db or not check_password_hash(
        users_db[username], password
    ):
        return jsonify({"msg": "Invalid username or password"}), 401
    access_token = create_access_token(identity=username)
    return jsonify(access_token=access_token), 200


@app.route("/api/strategies", methods=["GET", "POST"])
@jwt_required()
def strategies():
    if request.method == "GET":
        return jsonify(strategies_db), 200
    elif request.method == "POST":
        strategy = request.json
        strategies_db[strategy["name"]] = strategy
        return jsonify({"msg": "Strategy added successfully"}), 201


@app.route("/api/performance", methods=["GET"])
@jwt_required()
def performance():
    return jsonify(performance_db), 200


@app.route("/api/settings", methods=["GET", "POST"])
@jwt_required()
def settings():
    if request.method == "GET":
        user_id = get_jwt_identity()
        return jsonify(settings_db.get(user_id, {})), 200
    elif request.method == "POST":
        user_id = get_jwt_identity()
        settings_db[user_id] = request.json
        return jsonify({"msg": "Settings updated successfully"}), 201


@socketio.on("connect")
def handle_connect():
    print("Client connected")


@socketio.on("disconnect")
def handle_disconnect():
    print("Client disconnected")


if __name__ == "__main__":
    app = create_app()
    socketio.run(app, debug=True)
