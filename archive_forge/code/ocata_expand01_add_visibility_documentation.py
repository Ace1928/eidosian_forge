from alembic import op
from sqlalchemy import Column, Enum
from glance.cmd import manage
from glance.db import migration
from glance.db.sqlalchemy.schema import Boolean
add visibility to images

Revision ID: ocata_expand01
Revises: mitaka02
Create Date: 2017-01-27 12:58:16.647499

