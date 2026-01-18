import io
from ...migration import MigrationContext
from ...testing import assert_raises
from ...testing import config
from ...testing import eq_
from ...testing import is_
from ...testing import is_false
from ...testing import is_not_
from ...testing import is_true
from ...testing import ne_
from ...testing.fixtures import TestBase
def test_transaction_per_migration_sqlmode(self):
    context = self._fixture({'as_sql': True, 'transaction_per_migration': True})
    context.execute('step 1')
    with context.begin_transaction():
        context.execute('step 2')
        with context.begin_transaction(_per_migration=True):
            context.execute('step 3')
        context.execute('step 4')
    context.execute('step 5')
    if context.impl.transactional_ddl:
        self._assert_impl_steps('step 1', 'step 2', 'BEGIN', 'step 3', 'COMMIT', 'step 4', 'step 5')
    else:
        self._assert_impl_steps('step 1', 'step 2', 'step 3', 'step 4', 'step 5')