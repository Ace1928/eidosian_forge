from flask import Blueprint, jsonify
from flask_jwt_extended import jwt_required

api_bp = Blueprint("api", __name__)


@api_bp.route("/strategies", methods=["GET", "POST"])
@jwt_required()
def strategies():
    # Strategies logic
    return jsonify({"strategies": []}), 200


@api_bp.route("/performance", methods=["GET"])
@jwt_required()
def performance():
    # Performance logic
    return jsonify({"performance": []}), 200


@api_bp.route("/settings", methods=["GET", "POST"])
@jwt_required()
def settings():
    # Settings logic
    return jsonify({"settings": {}}), 200
