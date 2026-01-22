from dataclasses import dataclass
@dataclass
class VersionResponse:
    version: str
    ray_version: str
    ray_commit: str