import random
import numpy as np
import pygame
import sys


class GameEnvironment:
    def __init__(self, size=100):
        self.size = size
        self.grid = np.full((size, size), None, dtype=object)
        self.resources = np.zeros((size, size), dtype=int)
        self.resource_access_count = np.zeros((size, size), dtype=int)
        self.populate_resources()

    def populate_resources(self):
        for _ in range(self.size * 2):
            x, y = self.get_random_position()
            self.resources[x, y] += 1

    def place_creature(self, creature, position):
        x, y = position
        self.grid[x, y] = creature

    def move_creature(self, old_position, new_position):
        x_old, y_old = old_position
        x_new, y_new = new_position
        self.grid[x_new, y_new] = self.grid[x_old, y_old]
        self.grid[x_old, y_old] = None

    def get_random_position(self):
        while True:
            position = (
                random.randint(0, self.size - 1),
                random.randint(0, self.size - 1),
            )
            if self.grid[position] is None:
                return position

    def display(self):
        for row in self.grid:
            print(["." if cell is None else "C" for cell in row])
        print("\nResources:")
        for row in self.resources:
            print(row)
        print("\n")

    def update_resources(self, position):
        x, y = position
        self.resource_access_count[x, y] += 1
        if self.resource_access_count[x, y] >= 5:
            self.resources[x, y] = 0
            self.resource_access_count[x, y] = 0
            new_x, new_y = self.get_random_position()
            self.resources[new_x, new_y] += 1

    def apply_disease(self):
        for x in range(self.size):
            for y in range(self.size):
                if self.grid[x, y]:
                    creature = self.grid[x, y]
                    neighbors = sum(
                        1
                        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                        if 0 <= x + dx < self.size
                        and 0 <= y + dy < self.size
                        and self.grid[x + dx, y + dy]
                    )
                    if neighbors > 2 or self.resources[x, y] == 0:
                        creature.health -= 0.1


class Creature:
    def __init__(self, health, attack, speed, intelligence, strategy):
        self.health = health
        self.attack = attack
        self.speed = speed
        self.intelligence = intelligence  # Pathfinding intelligence factor
        self.strategy = strategy  # Strategy for encounter: 'battle', 'flee', 'group', 'mate', 'new_strategy'
        self.movement_energy_cost = 0.1 * self.speed  # Cost of movement based on speed

    def battle(self, other):
        while self.health > 0 and other.health > 0:
            other.health -= self.attack
            if other.health <= 0:
                return True
            self.health -= other.attack
        return self.health > 0

    def mutate(self):
        mutation_factor = 0.1
        self.health = max(1, self.health + random.uniform(-1, 1) * mutation_factor)
        self.attack = max(1, self.attack + random.uniform(-1, 1) * mutation_factor)
        self.speed = max(1, self.speed + random.uniform(-1, 1) * mutation_factor)
        self.intelligence = max(
            1, self.intelligence + random.uniform(-1, 1) * mutation_factor
        )
        if random.random() < 0.1:  # 10% chance to change strategy
            self.strategy = random.choice(
                ["battle", "flee", "group", "mate", "new_strategy"]
            )

    def __repr__(self):
        return f"Creature(Health: {self.health}, Attack: {self.attack}, Speed: {self.speed}, Intelligence: {self.intelligence}, Strategy: {self.strategy})"

    def choose_move(self, current_position, environment):
        x, y = current_position
        best_move = (x, y)
        best_score = float("-inf")

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < environment.size and 0 <= ny < environment.size:
                if environment.grid[nx, ny] is None:
                    score = (
                        random.uniform(0, 1) * self.intelligence
                        + environment.resources[nx, ny]
                    )
                    if score > best_score:
                        best_move = (nx, ny)
                        best_score = score

        return best_move

    def acquire_resources(self, position, environment):
        x, y = position
        if environment.resources[x, y] > 0:
            self.health += environment.resources[x, y]
            environment.update_resources((x, y))


class Battle:
    def __init__(self, environment):
        self.environment = environment

    def fight(self, position1, position2):
        creature1 = self.environment.grid[position1]
        creature2 = self.environment.grid[position2]

        if creature1 and creature2:
            winner = creature1.battle(creature2)
            if winner:
                self.environment.grid[position2] = creature1
            else:
                self.environment.grid[position1] = creature2

    def handle_encounter(self, position1, position2):
        creature1 = self.environment.grid[position1]
        creature2 = self.environment.grid[position2]

        if creature1 and creature2:
            if creature1.strategy == "battle" or creature2.strategy == "battle":
                self.fight(position1, position2)
            elif creature1.strategy == "flee" or creature2.strategy == "flee":
                if creature1.strategy == "flee":
                    new_position = creature1.choose_move(position1, self.environment)
                    self.environment.move_creature(position1, new_position)
                if creature2.strategy == "flee":
                    new_position = creature2.choose_move(position2, self.environment)
                    self.environment.move_creature(position2, new_position)
            elif creature1.strategy == "mate" and creature2.strategy == "mate":
                self.mate(creature1, creature2, position1)
            elif creature1.strategy == "group" or creature2.strategy == "group":
                self.group(creature1, creature2, position1)
            elif (
                creature1.strategy == "new_strategy"
                or creature2.strategy == "new_strategy"
            ):
                self.new_strategy(creature1, creature2, position1)

    def mate(self, creature1, creature2, position):
        new_health = (creature1.health + creature2.health) / 2
        new_attack = (creature1.attack + creature2.attack) / 2
        new_speed = (creature1.speed + creature2.speed) / 2
        new_intelligence = (creature1.intelligence + creature2.intelligence) / 2
        new_strategy = random.choice(
            ["battle", "flee", "group", "mate", "new_strategy"]
        )
        new_creature = Creature(
            new_health, new_attack, new_speed, new_intelligence, new_strategy
        )
        self.environment.place_creature(
            new_creature, self.environment.get_random_position()
        )

    def group(self, creature1, creature2, position):
        if creature1.strategy == "group" and creature2.strategy == "group":
            new_position1 = creature1.choose_move(position, self.environment)
            new_position2 = creature2.choose_move(position, self.environment)
            if new_position1 == new_position2:
                self.environment.move_creature(position, new_position1)
                self.environment.move_creature(position, new_position2)

    def new_strategy(self, creature1, creature2, position):
        # Implement new complex behavior here
        pass


class Evolution:
    def __init__(self, environment):
        self.environment = environment

    def evolve(self):
        new_creatures = []
        for x in range(self.environment.size):
            for y in range(self.environment.size):
                if self.environment.grid[x, y]:
                    creature = self.environment.grid[x, y]
                    creature.mutate()
                    new_creatures.append(creature)
        self.environment.grid = np.full(
            (self.environment.size, self.environment.size), None, dtype=object
        )
        for creature in new_creatures:
            self.environment.place_creature(
                creature, self.environment.get_random_position()
            )


class UserInteraction:
    def __init__(self, environment, evolution, battle):
        self.environment = environment
        self.evolution = evolution
        self.battle = battle

    def add_creature(self, health, attack, speed, intelligence, strategy, position):
        creature = Creature(health, attack, speed, intelligence, strategy)
        self.environment.place_creature(creature, position)

    def run_game(self, generations=10):
        pygame.init()
        screen_width, screen_height = (
            pygame.display.Info().current_w,
            pygame.display.Info().current_h,
        )
        cell_size = 5
        self.environment.size = min(screen_width, screen_height) // cell_size

        # Reinitialize the grid and resources after changing the size
        self.environment.grid = np.full(
            (self.environment.size, self.environment.size), None, dtype=object
        )
        self.environment.resources = np.zeros(
            (self.environment.size, self.environment.size), dtype=int
        )
        self.environment.resource_access_count = np.zeros(
            (self.environment.size, self.environment.size), dtype=int
        )
        self.environment.populate_resources()

        screen = pygame.display.set_mode(
            (screen_width, screen_height), pygame.RESIZABLE
        )
        clock = pygame.time.Clock()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            screen.fill((0, 0, 0))

            for generation in range(generations):
                self.evolution.evolve()
                self.environment.apply_disease()

                for x in range(self.environment.size):
                    for y in range(self.environment.size):
                        if self.environment.grid[x, y]:
                            creature = self.environment.grid[x, y]
                            new_position = creature.choose_move(
                                (x, y), self.environment
                            )
                            if new_position != (x, y):
                                self.environment.move_creature((x, y), new_position)

                for x in range(self.environment.size):
                    for y in range(self.environment.size):
                        if self.environment.grid[x, y]:
                            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                                nx, ny = x + dx, y + dy
                                if (
                                    0 <= nx < self.environment.size
                                    and 0 <= ny < self.environment.size
                                ):
                                    if self.environment.grid[nx, ny]:
                                        self.battle.handle_encounter((x, y), (nx, ny))

                for x in range(self.environment.size):
                    for y in range(self.environment.size):
                        if self.environment.grid[x, y]:
                            pygame.draw.rect(
                                screen,
                                (0, 255, 0),
                                (x * cell_size, y * cell_size, cell_size, cell_size),
                            )
                        elif self.environment.resources[x, y] > 0:
                            pygame.draw.rect(
                                screen,
                                (0, 0, 255),
                                (x * cell_size, y * cell_size, cell_size, cell_size),
                            )

            pygame.display.flip()
            clock.tick(5)  # Adjust the speed as needed


if __name__ == "__main__":
    environment = GameEnvironment()
    evolution = Evolution(environment)
    battle = Battle(environment)
    interaction = UserInteraction(environment, evolution, battle)

    # Add some initial creatures
    interaction.add_creature(10, 2, 3, 5, "battle", (0, 0))
    interaction.add_creature(12, 3, 2, 6, "flee", (4, 4))
    interaction.add_creature(8, 4, 1, 4, "mate", (2, 2))

    # Run the game indefinitely
    interaction.run_game(generations=1)
