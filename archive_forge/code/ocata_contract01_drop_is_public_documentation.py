from alembic import op
from sqlalchemy import Enum
from glance.cmd import manage
from glance.db import migration
remove is_public from images

Revision ID: ocata_contract01
Revises: mitaka02
Create Date: 2017-01-27 12:58:16.647499

