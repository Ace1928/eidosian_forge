from django.apps.registry import apps as global_apps
from django.db import migrations, router
from .exceptions import InvalidMigrationPlan
from .loader import MigrationLoader
from .recorder import MigrationRecorder
from .state import ProjectState
def _migrate_all_backwards(self, plan, full_plan, fake):
    """
        Take a list of 2-tuples of the form (migration instance, True) and
        unapply them in reverse order they occur in the full_plan.

        Since unapplying a migration requires the project state prior to that
        migration, Django will compute the migration states before each of them
        in a first run over the plan and then unapply them in a second run over
        the plan.
        """
    migrations_to_run = {m[0] for m in plan}
    states = {}
    state = self._create_project_state()
    applied_migrations = {self.loader.graph.nodes[key] for key in self.loader.applied_migrations if key in self.loader.graph.nodes}
    if self.progress_callback:
        self.progress_callback('render_start')
    for migration, _ in full_plan:
        if not migrations_to_run:
            break
        if migration in migrations_to_run:
            if 'apps' not in state.__dict__:
                state.apps
            states[migration] = state
            state = migration.mutate_state(state, preserve=True)
            migrations_to_run.remove(migration)
        elif migration in applied_migrations:
            migration.mutate_state(state, preserve=False)
    if self.progress_callback:
        self.progress_callback('render_success')
    for migration, _ in plan:
        self.unapply_migration(states[migration], migration, fake=fake)
        applied_migrations.remove(migration)
    last_unapplied_migration = plan[-1][0]
    state = states[last_unapplied_migration]
    for index, (migration, _) in enumerate(full_plan):
        if migration == last_unapplied_migration:
            for migration, _ in full_plan[index:]:
                if migration in applied_migrations:
                    migration.mutate_state(state, preserve=False)
            break
    return state