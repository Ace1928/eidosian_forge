import pygame
import random

# Initialize Pygame
pygame.init()

# Set screen size
screen_width = 800
screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))

# Define colors
white = (255, 255, 255)
black = (0, 0, 0)

# Create a list of particles
particles = []
for i in range(10):
    x = random.randint(0, screen_width)
    y = random.randint(0, screen_height)
    speed_x = random.uniform(-5, 5)
    speed_y = random.uniform(-5, 5)
    radius = random.randint(2, 10)
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    particles.append((x, y, speed_x, speed_y, radius, color))

# Main loop
while True:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

    # Update positions of each particle
    updated_particles = []
    for i, (x, y, speed_x, speed_y, radius, color) in enumerate(particles):
        x += speed_x
        y += speed_y
        
        # Check for collisions with the screen boundaries
        if x - radius < 0:
            x = radius
            speed_x *= -1
        elif x + radius > screen_width:
            x = screen_width - radius
            speed_x *= -1

        if y - radius < 0:
            y = radius
            speed_y *= -1
        elif y + radius > screen_height:
            y = screen_height - radius
            speed_y *= -1

        # Check for collisions between particles
        for j, (x2, y2, _, _, _, _) in enumerate(particles):
            if i != j:
                distance = ((x-x2)**2 + (y-y2)**2)**0.5
                if distance < radius * 2:
                    speed_x *= -1
                    speed_y *= -1
        updated_particles.append((x, y, speed_x, speed_y, radius, color))
    particles = updated_particles

    # Draw the background
    screen.fill(white)

    # Draw each particle
    for x, y, _, _, radius, color in particles:
        pygame.draw.circle(screen, color, (int(x), int(y)), radius)

    # Update display
    pygame.display.flip()

    # Wait a bit before updating again
    pygame.time.Clock().tick(60)