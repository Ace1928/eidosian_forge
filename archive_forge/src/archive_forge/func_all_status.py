from mlflow.protos.model_registry_pb2 import ModelVersionStatus as ProtoModelVersionStatus
@staticmethod
def all_status():
    return list(ModelVersionStatus._STATUS_TO_STRING.keys())