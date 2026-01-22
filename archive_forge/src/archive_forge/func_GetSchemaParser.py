from __future__ import absolute_import
import argparse
import json
import sys
import ruamel.yaml as yaml
from googlecloudsdk.third_party.appengine.admin.tools.conversion import yaml_schema_v1
from googlecloudsdk.third_party.appengine.admin.tools.conversion import yaml_schema_v1beta
def GetSchemaParser(api_version=None):
    return API_VERSION_SCHEMAS.get(api_version, yaml_schema_v1).SCHEMA