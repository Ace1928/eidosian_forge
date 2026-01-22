import logging
from alembic import op
drop_duplicate_killed_constraint

Revision ID: 0a8213491aaa
Revises: cfd24bdc0731
Create Date: 2020-01-28 15:26:14.757445

This migration drops a duplicate constraint on the `runs.status` column that was left as a byproduct
of an erroneous implementation of the `cfd24bdc0731_update_run_status_constraint_with_killed`
migration in MLflow 1.5. The implementation of this migration has since been fixed.
